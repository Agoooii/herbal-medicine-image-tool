"""Local product segmentation and mask-aware square compositing.

No image upload. Models are installed separately; a missing model is an error.
"""
from functools import lru_cache
from pathlib import Path
import os
import io
import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_MODELS = {'birefnet-general-lite', 'birefnet-general', 'u2net', 'isnet-general-use'}

@lru_cache(maxsize=2)
def model_session(model):
    if model not in SUPPORTED_MODELS:
        raise ValueError('不支持的本地模型: ' + model)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from rembg import new_session
    model_dir = Path(os.environ.get('U2NET_HOME', str(Path.home() / '.u2net')))
    if not (model_dir / (model + '.onnx')).is_file():
        raise RuntimeError(f'本地模型缺失: {model_dir / (model + ".onnx")}。请先运行安装抠图模型.py；本次不使用旧算法替代。')
    return new_session(model, providers=['CPUExecutionProvider'])


def mask_bbox(mask, threshold=8):
    points = cv2.findNonZero((mask > threshold).astype(np.uint8))
    return cv2.boundingRect(points) if points is not None else None


def guided_alpha(rgb, alpha, radius=4):
    """Small-radius guided filtering only near the silhouette, not inside the bag."""
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    p = alpha.astype(np.float32) / 255
    box = lambda x: cv2.boxFilter(x, -1, (2 * radius + 1,) * 2, borderType=cv2.BORDER_REFLECT)
    mean_i, mean_p = box(guide), box(p)
    a = (box(guide * p) - mean_i * mean_p) / (box(guide * guide) - mean_i * mean_i + 0.001)
    b = mean_p - a * mean_i
    q = np.clip(box(a) * guide + box(b), 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    hard = (p >= 0.5).astype(np.uint8)
    band = cv2.dilate(hard, kernel) != cv2.erode(hard, kernel)
    result = np.where(band, q, p)
    result[result < 0.02] = 0
    result[result > 0.98] = 1
    return np.round(result * 255).astype(np.uint8)


def clean_product_mask(alpha, fill_holes=True):
    """Keep the single packaged object without convexifying its folds or corners."""
    binary = (alpha >= 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise ValueError('模型未找到产品主体')
    sizes = stats[1:, cv2.CC_STAT_AREA]
    main_id = int(np.argmax(sizes)) + 1
    main = (labels == main_id).astype(np.uint8)
    warnings = []
    if len(sizes) > 1 and np.sort(sizes)[-2] > sizes.max() * 0.1:
        warnings.append('模型检出多个分离主体，请检查是否漏掉包装部分')
    if fill_holes:
        # Exterior contours fill enclosed false background, but preserve exterior concavities.
        contours, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(main, contours, -1, 1, cv2.FILLED)
    support = cv2.dilate(main, np.ones((9, 9), np.uint8))
    result = np.where(support > 0, alpha, 0).astype(np.uint8)
    if fill_holes:
        # Package contents must be opaque, regardless of their similarity to the background.
        interior = cv2.erode(main, np.ones((3, 3), np.uint8))
        result[interior > 0] = 255
    return result, warnings


def segment_product(img, model='birefnet-general-lite', fill_holes=True, refine=True):
    from rembg import remove
    work = img.convert('RGB').copy()
    work.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    alpha = np.asarray(remove(work, session=model_session(model), only_mask=True).convert('L'))
    alpha, warnings = clean_product_mask(alpha, fill_holes=fill_holes)
    used_context = False
    initial_coverage = float(np.mean(alpha >= 127))
    if warnings or initial_coverage < 0.18:
        # A tight photograph can make a saliency model select the label alone.
        # Supply neutral surrounding context ONLY to inference; keep original image pixels.
        arr = np.asarray(work)
        corner_samples = np.concatenate((arr[:10, :10].reshape(-1, 3),
                                         arr[:10, -10:].reshape(-1, 3),
                                         arr[-10:, :10].reshape(-1, 3),
                                         arr[-10:, -10:].reshape(-1, 3)))
        bg = tuple(int(v) for v in np.median(corner_samples, axis=0))
        pad = round(max(work.size) * 0.15)
        context = ImageOps.expand(work, pad, fill=bg)
        retry = remove(context, session=model_session(model), only_mask=True).convert('L')
        retry = np.asarray(retry.crop((pad, pad, pad + work.width, pad + work.height)))
        retry, retry_warnings = clean_product_mask(retry, fill_holes=fill_holes)
        candidate_coverage = float(np.mean(retry >= 127))
        retention = np.sum((retry >= 127) & (alpha >= 127)) / max(1, np.sum(alpha >= 127))
        if (not retry_warnings and retention >= 0.8
                and max(0.12, initial_coverage * 1.2) <= candidate_coverage < 0.97):
            alpha = retry
            warnings = ['初次分割不完整，已增加背景上下文重新识别，请复核袋边']
            used_context = True
    if refine:
        alpha = guided_alpha(np.asarray(work), alpha)
    coverage = float(np.mean(alpha >= 127))
    if coverage < 0.12:
        warnings.append('主体占比过小，可能只识别到了标签，或拍摄距离过远')
    if coverage > 0.97:
        warnings.append('几乎整张图被判为主体，请检查背景残留')
    border = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
    if np.mean(border > 127) > 0.015:
        warnings.append('主体触及原图边界，请检查原图是否已截断袋角')
    if work.size != img.size:
        alpha = np.asarray(Image.fromarray(alpha).resize(img.size, Image.Resampling.LANCZOS))
    bbox = mask_bbox(alpha)
    if bbox is None:
        raise ValueError('精修后没有有效主体')
    return alpha, bbox, {'model': model, 'coverage': round(coverage, 4), 'context_retry': used_context, 'warnings': warnings}


def compose_square(img, mask, size=1200, pad=30, erode=0, feather=0):
    """Use alpha for the crop; preserve pale tips and resize premultiplied colors."""
    size, pad = int(size), int(pad)
    if size < 32 or pad < 0 or pad * 2 >= size:
        raise ValueError('输出像素和留白参数无效')
    if mask.shape != (img.height, img.width):
        raise ValueError('蒙版尺寸与原图不一致')
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError('空蒙版不能生成商品图')
    x, y, w, h = bbox
    rgba = img.convert('RGBA').crop((x, y, x + w, y + h))
    rgba.putalpha(Image.fromarray(mask).crop((x, y, x + w, y + h)))
    scale = (size - 2 * pad) / max(w, h)
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgba = rgba.convert('RGBa').resize(target, Image.Resampling.LANCZOS).convert('RGBA')
    a = np.array(rgba.getchannel('A'))
    # Optional legacy erosion/feather are measured in FINAL output pixels.
    if erode > 0:
        radius = int(round(erode))
        a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2))
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(feather))
    rgba.putalpha(Image.fromarray(a))
    out = Image.new('RGB', (size, size), 'white')
    out.paste(rgba, ((size - target[0]) // 2, (size - target[1]) // 2), rgba)
    return out


def encode_square(img, max_bytes, q_init=92, q_min=60):
    """Maintain required dimensions; never silently return an oversized JPEG."""
    qualities = list(range(q_init, q_min - 1, -4))
    if not qualities or qualities[-1] != q_min:
        qualities.append(q_min)
    for quality in qualities:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
    raise ValueError('在保持指定像素和最低画质时无法达到文件体积上限，请调整 max_kb 或 q_min')
