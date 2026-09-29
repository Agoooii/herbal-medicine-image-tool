"""
图片批量处理 v8

用法：图片命名好 → 按类型放进对应文件夹 → 双击运行 → 取结果

检测方法（在 folders.ini 配置）：
  bag / product — BiRefNet Lite + 完整包装保护 + 断袋重试（产品图默认）
  board — 整块标尺板外边界检测（标尺图默认）
  threshold — 旧灰度阈值，仅保留兼容
  adaptive / color / bag_legacy — 保留的旧检测方法
  rembg — 旧 ISNet 模型接口

Debug 模式：设置 DEBUG=True 可保存每一步的中间结果到 调试/ 目录

依赖：pip install -r requirements.txt；首次换电脑运行 安装抠图模型.py
"""

# ══════════════════════════════════════════════
# 抠图默认参数
# ══════════════════════════════════════════════

PAD      = 30
FEATHER  = 5
ERODE    = 10
BLACK_TH = 45

# 输出规格
CUTOUT_MB   = 3
CUTOUT_W    = 800
COMPRESS_MB = 3
COMPRESS_W  = 1500
Y1C_SIZE    = 800
Y1C_KB      = 2000

Q_INIT = 92
Q_MIN  = 60

# Debug 模式：设为 True 会把中间步骤保存到 调试/ 目录
DEBUG = False

# ══════════════════════════════════════════════

import cv2, numpy as np, io, shutil, json
from pathlib import Path
from PIL import Image, ExifTags, ImageOps
from product_cutout import segment_product, compose_square, encode_square
from board_cutout import segment_board, orient_board
from datetime import datetime

# OpenCV 连通分量统计字段下标（cv2 某些版本不暴露命名常量，引用数字更稳）
CC_LEFT, CC_TOP, CC_W, CC_H, CC_AREA = 0, 1, 2, 3, 4

BASE = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "folders.ini"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def log(msg, level="info"):
    tag = {"info": "  ", "ok": "OK ", "warn": "!! ", "err": "XX "}[level]
    print(f"[{datetime.now():%H:%M:%S}] {tag}{msg}")


def save_debug(img_arr, name, prefix=""):
    """保存调试用的中间结果图片到 调试/ 目录。img_arr 为 BGR numpy 数组或 None。"""
    if not DEBUG or img_arr is None:
        return
    debug_dir = Path(__file__).resolve().parent / "调试"
    debug_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{name}.jpg" if prefix else f"{name}.jpg"
    cv2.imwrite(str(debug_dir / fname), img_arr)
    log(f"debug: {fname}", "info")


def imgs(folder):
    p = Path(folder)
    return sorted([f for f in p.iterdir()
                   if f.is_file() and f.suffix.lower() in EXT]) if p.exists() else []


def subdirs(folder):
    """返回包含图片的子文件夹列表，没有则返回空列表"""
    p = Path(folder)
    if not p.exists():
        return []
    result = []
    for d in sorted(p.iterdir()):
        if d.is_dir() and d.name != "已完成" and imgs(d):
            result.append(d)
    return result


# ══════════════════════════════════════════════
# Adaptive 预处理：Gamma 校正暗部提亮
# ══════════════════════════════════════════════

def apply_gamma_for_adaptive(cv_img):
    """四角亮度偏高时，Gamma 校正压暗背景、提亮主体。
    返回校正后的 BGR 图像。"""
    h, w = cv_img.shape[:2]
    cw2, ch2 = min(20, w // 4), min(20, h // 4)
    c_sample = np.vstack([
        cv_img[0:ch2, 0:cw2],
        cv_img[0:ch2, -cw2:],
        cv_img[-ch2:, 0:cw2],
        cv_img[-ch2:, -cw2:]
    ]).reshape(-1, 3).astype(np.float32)
    if np.mean(c_sample) > 15:
        gamma = max(0.75, 1.0 - np.mean(c_sample) / 60)
        lut = (255.0 * (np.arange(256) / 255.0) ** gamma).clip(0, 255).astype(np.uint8)
        return cv2.LUT(cv_img, lut)
    return cv_img


# ══════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════

DEFAULT_CONFIG = {
    "抠图模式": {
        "folders": [
            {"name": "药师帮_标尺图",     "method": "board", "erode": 0, "pad": 30, "feather": 0, "size": 1200, "max_kb": 3072},
            {"name": "药师帮_产品图", "method": "bag",    "erode": 0, "pad": 30, "feather": 0},
        ],
        "description": "抠图-白底-1:1-压缩"
    },
    "压缩模式": {
        "folders": [{"name": "只需压缩的放这里"}],
        "description": "只压缩，原比例"
    },
    "1药城模式": {
        "folders": [{"name": "1药城_800x800"}],
        "description": "800x800 + 2MB"
    }
}


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            log(f"已加载: {CONFIG_FILE.name}")
            return cfg
        except Exception as e:
            log(f"配置损坏: {e}，用默认", "warn")
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    log(f"已创建: {CONFIG_FILE.name}")
    return DEFAULT_CONFIG


# ══════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════

def rotate(img):
    # Apply orientation (including mirrored EXIF variants) before converting to RGB.
    return ImageOps.exif_transpose(img)


def compress(img, max_bytes, q_init=92, q_min=60, min_w=1500):
    q = q_init
    while q >= q_min:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        if len(buf.getvalue()) <= max_bytes: return buf.getvalue()
        q -= 5
    scale = 0.95
    while True:
        nw = max(min_w, int(img.width * scale))
        nh = int(img.height * nw / img.width)
        rs = img.resize((nw, nh), Image.LANCZOS)
        buf = io.BytesIO()
        rs.save(buf, "JPEG", quality=q_min, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes or nw <= min_w: return data
        scale *= 0.95


# ══════════════════════════════════════════════
# 主体检测（三种方法）
# ══════════════════════════════════════════════

def _small_component_repair(mask, max_area=200):
    """小连通块修复：把 mask 内的孤立小黑块（面积<=max_area）填平，
    同时也填 bg 内的孤立小亮块（袋装图的反光黑洞常以这种形式出现）。
    """
    # 填前景内小洞（黑块）：开运算天然会删小洞，先用闭运算填上
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # 然后数连通分量，删掉面积过小的"噪点前景"（避免袋角碎块）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, CC_AREA] >= max_area // 2:
            out[labels == i] = 255
    return out


def _edge_shadow_suppress(mask, cv_img, bg_median, edge_px=10,
                          shadow_ratio=1.45, min_keep=0.75, bg_tol=1.15):
    """=== B段核心：边缘阴影抑制（替代 GrabCut）===

    问题：mask 边缘一圈常混入"袋子投在桌面/地面上的阴影"，表现为
          - 紧贴 mask 边缘
          - 颜色和背景同调，只是更暗（阴影 ≈ 背景 × 衰减系数）
    做法：
      1) 距离变换找出"距前景边缘 <= edge_px"的缓冲区
      2) 缓冲区内，找颜色接近背景（欧氏距离 < 背景自身 MAD 的若干倍）的像素
      3) 这些像素从前景剔除
      4) 若剔除后主体面积掉到 min_keep 以下，说明误伤，回退原 mask

    返回精修后的 mask。
    """
    if mask is None or not mask.any():
        return mask

    mask = mask.copy()
    # 1) 距离变换：前景内每个像素到最近背景的距离
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    # 边缘缓冲区 = 前景 且 距边缘 <= edge_px
    edge_band = (mask > 0) & (dist <= edge_px)
    if not edge_band.any():
        return mask

    # 2) 背景色距离图（用角点 MAD 做自适应容差，避免纯色背景上误伤主体）
    diff = np.sqrt(np.sum((cv_img.astype(np.float32) - bg_median) ** 2, axis=2))

    # 背景自身波动范围：用四角区域 MAD 近似（这里退化为一个保守常数带）
    # 阴影通常比背景暗 15%-50%，颜色仍接近，故容差不宜过大
    bg_noise = max(18.0, 0.28 * np.mean(np.abs(bg_median)))
    th_bg = bg_noise * shadow_ratio

    near_bg = diff < th_bg

    # 3) 只剔除"边缘缓冲区内 且 接近背景"的像素
    remove = edge_band & near_bg

    if not remove.any():
        return mask

    candidate = mask.copy()
    candidate[remove] = 0

    # 4) 回退保护：面积掉太多就放弃这次精修
    area_before = int(np.sum(mask > 0))
    area_after = int(np.sum(candidate > 0))
    if area_after < min_keep * area_before:
        if DEBUG:
            log(f"edge_suppress: 面积掉 {(1-area_after/area_before)*100:.0f}%，回退",
                "warn")
        return mask

    # 剔除后再做一次小块修复，把阴影剔除造成的毛边收一收
    candidate = _small_component_repair(candidate, max_area=150)
    return candidate


def find_subject(img, th=45, method="threshold"):
    """
    threshold: 灰度阈值 — 像素灰度>阈值=主体（标尺图）
    color:     三通道分别判断 — R>th 或 G>th 或 B>th = 主体（袋装图）
    adaptive:  多线索融合 — HSV饱和度+RGB+边缘约束（袋装图推荐）
    bag/product: BiRefNet Lite + 包装完整性保护
    rembg:     ISNet 模型（需本地 isnet-general-use.onnx）
    """
    if method == "board":
        mask, bbox, report = segment_board(img)
        for warning in report["warnings"]:
            log(warning, "warn")
        return mask, bbox
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # ---- removebg API ----
    if method == "removebg_api":
        import requests as _req
        api_key = globals().get("_removebg_key", "")
        if not api_key:
            log("removebg API key未配置", "err")
            return None, None

        try:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            resp = _req.post(
                "https://api.remove.bg/v1.0/removebg",
                files={"image_file": ("image.png", buf.getvalue(), "image/png")},
                data={"size": "regular"},
                headers={"X-Api-Key": api_key},
                timeout=30
            )
            if resp.status_code != 200:
                log(f"removebg API失败: {resp.status_code} {resp.text[:100]}", "err")
                return None, None
        except Exception as e:
            log(f"removebg API失败: {e}", "err")
            return None, None

        rgba = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        alpha = np.array(rgba.split()[3])
        fg = alpha > 10
        if not fg.any():
            return None, None
        rows, cols = np.where(fg)
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        if (x2-x1)*(y2-y1) < 0.03 * w * h:
            return None, None
        return alpha, (x1, y1, x2-x1, y2-y1)

    # Local AI product segmentation. Legacy algorithms remain explicitly selectable.
    if method in {"bag", "product", "rembg"}:
        model = "isnet-general-use" if method == "rembg" else "birefnet-general-lite"
        mask, bbox, report = segment_product(img, model=model, fill_holes=method != "rembg")
        for warning in report["warnings"]:
            log(warning, "warn")
        return mask, bbox

    # ---- greenscreen: 灰度边缘检测（不依赖颜色）----
    if method == "greenscreen":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 90)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 0.03 * w * h:
            return None, None
        hull = cv2.convexHull(best)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        x, y, bw, bh = cv2.boundingRect(hull)
        return mask, (x, y, bw, bh)

    # ---- adaptive: 通用背景检测法（四角取样 + 颜色距离 + 连通分量）----
    if method == "adaptive":
        corner_w = min(20, w // 4)
        corner_h = min(20, h // 4)
        corners = np.vstack([
            cv_img[0:corner_h, 0:corner_w].reshape(-1, 3),
            cv_img[0:corner_h, w-corner_w:w].reshape(-1, 3),
            cv_img[h-corner_h:h, 0:corner_w].reshape(-1, 3),
            cv_img[h-corner_h:h, w-corner_w:w].reshape(-1, 3),
        ]).astype(np.float32)

        bg_median = np.median(corners, axis=0)
        bg_std    = np.std(corners, axis=0)

        if DEBUG:
            log(f"adaptive: bg_color=({bg_median[0]:.0f},{bg_median[1]:.0f},{bg_median[2]:.0f}) "
                f"std=({bg_std[0]:.0f},{bg_std[1]:.0f},{bg_std[2]:.0f})")

        diff = np.sqrt(np.sum((cv_img.astype(np.float32) - bg_median) ** 2, axis=2))
        bg_mad = np.mean(np.abs(corners - bg_median))

        # === A6补丁：浅色背景场景专用分支 ===
        # 场景：浅色半透明袋装 + 灰/浅色织物背景（如 秦艽、黄芪 实拍）
        # 问题：背景本身偏亮（如 148,133,116 灰黄），袋子透明区/文字高光与背景
        #       颜色距离很近。若沿用深色场景的宽 threshold，袋子边缘会被拉远、
        #       袋内透明区被判成背景 → 标签被掏空。
        # 判据：背景中位亮度 > 115 且饱和度低（灰度感背景）→ 走"浅色分支"
        bg_hsv = cv2.cvtColor(bg_median.reshape(1, 1, 3).astype(np.uint8),
                              cv2.COLOR_BGR2HSV)[0, 0]
        bg_val = int(bg_hsv[2])
        bg_sat = int(bg_hsv[1])
        is_light_bg = bg_val > 115

        if is_light_bg:
            # === B2: 浅色场景自适应分支（KMeans 可信背景聚类）===
            # 问题：工厂背景多为渐变灰/浅色织物，单一四角中位色无法覆盖暗部背景；
            #       而直接用四边条带中位数又会被贴边/贴底的袋子污染。
            # 思路：
            #   1. 用四角中位色初筛"可信背景像素"；
            #   2. 对这些像素做 KMeans 聚类（k=3），分离背景、袋子、内容物等色心；
            #   3. 按"色心是否贴到图像边缘"投票，选出真正的背景色心；
            #   4. 用像素到最近背景色心的距离 + MAD 自适应阈值做二值化。
            # 这样即使袋子贴底/贴边，袋子对应的色心因不贴边（或少贴边）而被排除。

            diff0 = np.sqrt(np.sum((cv_img.astype(np.float32) - bg_median) ** 2, axis=2))

            # 初筛可信背景样本：只取与四角中位色接近的像素
            bg_mad = np.mean(np.abs(corners - bg_median))
            init_th = max(25, min(65, bg_mad * 4 + 25))
            sample_mask = diff0 < init_th
            samples = cv_img[sample_mask].reshape(-1, 3).astype(np.float32)

            K_KMEANS = 3
            MAX_SAMPLES = 60000
            use_kmeans = len(samples) >= K_KMEANS

            if use_kmeans and len(samples) > MAX_SAMPLES:
                np.random.seed(42)
                idx = np.random.choice(len(samples), MAX_SAMPLES, replace=False)
                samples = samples[idx]

            if use_kmeans:
                print(f"[B2] kmeans branch, samples={len(samples)}, init_th={init_th:.1f}")
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
                _, _, centers = cv2.kmeans(
                    samples, K_KMEANS, None, criteria, 8, cv2.KMEANS_PP_CENTERS
                )

                # 全图 label map：每个像素归属哪个色心
                label_map = np.zeros((h, w), dtype=np.int32)
                dist_map = np.full((h, w), np.inf, dtype=np.float32)
                for i, c in enumerate(centers):
                    d = np.sqrt(np.sum((cv_img.astype(np.float32) - c) ** 2, axis=2))
                    update = d < dist_map
                    label_map[update] = i
                    dist_map[update] = d[update]

                # 边缘投票：真正的背景色心应在多条边上都有显著分布
                side_scores = []
                for i in range(K_KMEANS):
                    m = (label_map == i)
                    side_scores.append([
                        m[0, :].mean(),
                        m[-1, :].mean(),
                        m[:, 0].mean(),
                        m[:, -1].mean(),
                    ])

                # 选在至少两条边占比超过 8% 的色心作为背景
                bg_labels = [i for i, scores in enumerate(side_scores)
                             if sum(1 for s in scores if s > 0.08) >= 2]
                # 兜底：若未选出，选四边平均占比最高的那个
                if not bg_labels:
                    avg_scores = [np.mean(s) for s in side_scores]
                    bg_labels = [int(np.argmax(avg_scores))]

                bg_centers = centers[bg_labels].astype(np.float32)
                print(f"[B2] bg_labels={bg_labels}, bg_centers={bg_centers}")

                # 像素到最近背景色心的距离
                diff = np.full((h, w), np.inf, dtype=np.float32)
                for c in bg_centers:
                    d = np.sqrt(np.sum((cv_img.astype(np.float32) - c) ** 2, axis=2))
                    np.minimum(diff, d, out=diff)

                # 自适应 threshold：在可信背景区域（sample_mask）的 diff 分布上取 median + 4*MAD
                bg_vals = diff[sample_mask]
                if len(bg_vals) > 0:
                    med = np.median(bg_vals)
                    mad = np.median(np.abs(bg_vals - med))
                    threshold = max(22, min(55, med + 4 * mad))
                else:
                    threshold = 35
                print(f"[B2] med={med:.1f} mad={mad:.1f} threshold={threshold:.1f}")
            else:
                # 样本不足，fallback 到原四角中位色
                diff = diff0
                threshold = max(25, min(60, bg_mad * 4 + 25))
        else:
            # 深色分支（原逻辑）：让"非纯黑但偏暗"背景也能拉出主体
            threshold = max(15, min(80, bg_mad * 6 + 20))

        if DEBUG:
            log(f"adaptive: bg_MAD={bg_mad:.1f} val={bg_val} sat={bg_sat} "
                f"{'浅色' if is_light_bg else '深色'}分支 threshold={threshold:.0f}")

        bg_candidate = diff < threshold

        if DEBUG:
            save_debug((bg_candidate * 255).astype(np.uint8), "A1_bg_candidate", "")

        bg_bin = (bg_candidate * 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bg_bin, connectivity=4
        )

        edge_labels = set()
        edge_labels.update(labels[0, :].tolist())
        edge_labels.update(labels[-1, :].tolist())
        edge_labels.update(labels[:, 0].tolist())
        edge_labels.update(labels[:, -1].tolist())
        edge_labels.discard(0)

        if not edge_labels:
            if DEBUG:
                log("adaptive: 无边触背景，回退OTSU", "warn")
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            otsu_th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_th = max(otsu_th, 40)
            corner_v = np.mean(cv2.cvtColor(corners.reshape(-1,1,3).astype(np.uint8),
                                             cv2.COLOR_BGR2HSV)[:, :, 2])
            if corner_v < 80:
                _, fg_bin = cv2.threshold(gray, otsu_th, 255, cv2.THRESH_BINARY)
            else:
                _, fg_bin = cv2.threshold(gray, otsu_th, 255, cv2.THRESH_BINARY_INV)
        else:
            bg_mask = np.isin(labels, list(edge_labels))
            fg_bin = (~bg_mask * 255).astype(np.uint8)

            # === A1补丁：Sobel 边缘引导的闭运算，保留袋装边缘纹理 ===
            # 思路：先算出"主体内"的边缘图（Sobel），闭运算时只填非边缘区域，
            #       避免 21×21 大核把袋边褶皱、文字、袋角一起填平
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(sobel_x, sobel_y)
            edge_norm = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edge_bin = cv2.threshold(edge_norm, 28, 255, cv2.THRESH_BINARY)

            # 只在"低边缘"区域闭运算（填洞），高边缘区域保持原样
            close_mask = cv2.morphologyEx(
                fg_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
            )
            # 边缘区域还回 fg_bin 的原值
            fg_bin = np.where(edge_bin > 0, fg_bin, close_mask)

            if DEBUG:
                save_debug(edge_bin, "A1b_edge_bin", "")
                save_debug(fg_bin, "A1c_after_guided_close", "")

            # 轻量开运算，去掉小的孤立噪点（核 3×3 而非 7×7，保留袋角折痕）
            fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            # === A3补丁：小连通块修复（填袋装内的反光黑洞/小黑点）===
            # 反向：把 fg 内的"小黑块"（连通且面积小）和"小黑洞"（bg 内的小连通）互相修补
            fg_bin = _small_component_repair(fg_bin, max_area=200)

        if DEBUG:
            save_debug(fg_bin, "A2_fg_mask", "")

        cnts, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 0.03 * w * h:
            return None, None

        if DEBUG:
            hull_viz = cv_img.copy()
            cv2.drawContours(hull_viz, [best], -1, (0, 255, 0), 3)
            save_debug(hull_viz, "A3_contour", "")

        # 直接使用清理后的前景蒙版（不用凸包，避免黑底图上包入过多背景）
        mask = fg_bin
        x, y, bw, bh = cv2.boundingRect(best)
        return mask, (x, y, bw, bh)

    # ---- bag: 袋装专用——adaptive 粗分割 + 边缘阴影抑制精修（A+B 融合）----
    # 说明：原方案用 GrabCut 精修，但在"浅色半透明袋 + 灰黄背景"场景下不稳定
    #       （GrabCut 会把深色内容物/浅色袋面判为背景，导致袋子被切成两半）。
    #       现改为：adaptive（含浅色/深色自适应分支）+ 边缘阴影抑制。
    #       优势：秒级完成、无外部依赖、对半透明袋更友好。
    if method == "bag_legacy":
        # Step 1: adaptive 粗分（内部已含浅色/深色分支）
        mask_a, bbox_a = find_subject(img, th=th, method="adaptive")
        if mask_a is None:
            return None, None

        # Step 2: 复算背景中位色，供阴影抑制使用
        corner_w = min(20, w // 4)
        corner_h = min(20, h // 4)
        corners = np.vstack([
            cv_img[0:corner_h, 0:corner_w].reshape(-1, 3),
            cv_img[0:corner_h, w-corner_w:w].reshape(-1, 3),
            cv_img[h-corner_h:h, 0:corner_w].reshape(-1, 3),
            cv_img[h-corner_h:h, w-corner_w:w].reshape(-1, 3),
        ]).astype(np.float32)
        bg_median = np.median(corners, axis=0)

        # Step 3: 边缘阴影抑制
        # 边缘带宽度随图像尺寸自适应（大图用更宽的带，小图收窄）
        edge_px = int(max(6, min(18, min(w, h) * 0.006)))
        mask_b = _edge_shadow_suppress(
            mask_a, cv_img, bg_median,
            edge_px=edge_px,
            shadow_ratio=1.45,
            min_keep=0.75,
        )

        if DEBUG:
            save_debug(mask_b, "B2_bag_mask", "")

        # Step 4: 收尾——小块修复 + 轻量闭运算（保边）
        mask_b = _small_component_repair(mask_b, max_area=200)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Step 5: 结果校验
        cnts, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            log("bag: 精修后无主体，回退 adaptive", "warn")
            return mask_a, bbox_a
        best_b = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best_b) < 0.03 * w * h:
            log("bag: 精修后主体过小，回退 adaptive", "warn")
            return mask_a, bbox_a

        xb, yb, bwb, bhb = cv2.boundingRect(best_b)
        return mask_b, (xb, yb, bwb, bhb)

    # ---- color: 三通道分别判断 ----
    if method == "color":
        B, G, R = cv_img[:,:,0], cv_img[:,:,1], cv_img[:,:,2]
        mask_arr = (R > th) | (G > th) | (B > th)
        mask_bin = (mask_arr * 255).astype(np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None, None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 0.05*w*h: return None, None
        hull = cv2.convexHull(best)
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        x, y, bw, bh = cv2.boundingRect(hull)
        return mask, (x, y, bw, bh)

    # ---- threshold（默认）----
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    ot, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    use_th = ot if 40 < ot < 200 else th
    _, bin_img = cv2.threshold(gray, use_th, 255, cv2.THRESH_BINARY)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.05*w*h: return None, None
    hull = cv2.convexHull(best)
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    x, y, bw, bh = cv2.boundingRect(hull)
    return mask, (x, y, bw, bh)


# ══════════════════════════════════════════════
# 白底 + 1:1
# ══════════════════════════════════════════════

def white_bg(img, mask, bbox, pad=30, feather=1.5, erode=2):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    x, y, bw, bh = bbox
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(w, x+bw+pad), min(h, y+bh+pad)
    crop = cv_img[y1:y2, x1:x2]
    mc = mask[y1:y2, x1:x2]

    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode*2+1, erode*2+1))
        mc = cv2.erode(mc, k)

    if feather > 0:
        a = cv2.GaussianBlur(mc.astype(np.float32), (0,0), feather) / 255.0
    else:
        a = mc.astype(np.float32) / 255.0

    a3 = np.stack([np.clip(a,0,1)]*3, axis=2)
    out = crop.astype(np.float32)*a3 + 255.0*(1-a3)
    return Image.fromarray(cv2.cvtColor(out.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))


def square(img, pad=30):
    arr = np.array(img)
    # === A4补丁：容忍外圈≤5%面积的浅色/反光残留，避免把袋子边缘吃掉一圈 ===
    # 用"非白(<252)"先粗查主体，再做一次轻微腐蚀剔除极薄的浅边
    nw = ~np.all(arr > 248, axis=2)
    if nw.any():
        rs, cs = np.any(nw, axis=1), np.any(nw, axis=0)
        y1, y2 = int(np.where(rs)[0][0]), int(np.where(rs)[0][-1])
        x1, x2 = int(np.where(cs)[0][0]), int(np.where(cs)[0][-1])

        # 二次精修：每条边向内扫描，碰到的反光浅条>pad*0.3 才认作主体
        h, w = arr.shape[:2]
        pad2 = max(2, int(pad * 0.3))
        # 上
        while y1 < y2 - 10 and np.mean(~np.all(arr[y1] > 248, axis=1)) < 0.05:
            y1 += 1
        # 下
        while y2 > y1 + 10 and np.mean(~np.all(arr[y2] > 248, axis=1)) < 0.05:
            y2 -= 1
        # 左
        while x1 < x2 - 10 and np.mean(~np.all(arr[:, x1] > 248, axis=1)) < 0.05:
            x1 += 1
        # 右
        while x2 > x1 + 10 and np.mean(~np.all(arr[:, x2] > 248, axis=1)) < 0.05:
            x2 -= 1
    else:
        y1, y2, x1, x2 = 0, img.height-1, 0, img.width-1
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(img.width-1, x2+pad), min(img.height-1, y2+pad)
    tight = img.crop((x1, y1, x2+1, y2+1))
    side = max(tight.size)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(tight, ((side - tight.width) // 2, (side - tight.height) // 2))
    return sq


def needs_review(sq_img=None, mask=None, orig_w=1, orig_h=1,
                 min_product_pct=20, min_mask_pct=25,
                 _bbox=None):
    """检查产品是否残缺
    - 输出图非白占比 < min_product_pct → 大面积丢失
    - 原图蒙版覆盖率 < min_mask_pct → 产品被部分吃掉
    - === A5补丁 === 边缘连续性差 + 袋子断两半 → 报警
    边缘连续性: 取 mask 边界最长一段 / 总周长 < 0.4 → 边缘破碎
    断袋检测: 主体连通分量有 ≥2 个，前两大各占 >10%、中间 >30% 是背景
    """
    if sq_img is not None:
        arr = np.array(sq_img)
        non_white = np.sum(np.max(arr, axis=2) < 240)
        pct = non_white / (sq_img.width * sq_img.height) * 100
        if pct < min_product_pct:
            return True

    if mask is None:
        return False

    mask_pct = np.sum(mask > 0) / (orig_w * orig_h) * 100
    if mask_pct < min_mask_pct:
        return True

    # === A5-1: 边缘连续性检测 ===
    edges = cv2.Canny(mask, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        total_perim = sum(cv2.arcLength(c, True) for c in cnts)
        max_perim = max(cv2.arcLength(c, True) for c in cnts)
        if total_perim > 0 and max_perim / total_perim < 0.4:
            # 边缘破碎：提示让用户检查（很可能是袋装断裂/反光误识别）
            return True

    # === A5-2: 断袋检测 ===
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if num >= 3:  # 至少 2 个前景块 + 1 个背景
        # 找前两大前景块
        sizes_with_id = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num)]
        sizes_with_id.sort(reverse=True)
        if len(sizes_with_id) >= 2 and \
           sizes_with_id[0][0] > 0.15 * mask.size and \
           sizes_with_id[1][0] > 0.1 * mask.size:
            # 检查这两块是否被背景隔开（bbox 中间区域有 >40% 是 bg）
            id1 = sizes_with_id[0][1]
            id2 = sizes_with_id[1][1]
            x1a, y1a = stats[id1, CC_LEFT], stats[id1, CC_TOP]
            wa, ha = stats[id1, CC_W], stats[id1, CC_H]
            x1b, y1b = stats[id2, CC_LEFT], stats[id2, CC_TOP]
            wb, hb = stats[id2, CC_W], stats[id2, CC_H]
            # 取两块 bbox 的"中间条带"（水平方向或垂直方向任一即可）
            mid_y = (max(y1a, y1b) + min(y1a + ha, y1b + hb)) // 2
            mid_x = (max(x1a, x1b) + min(x1a + wa, x1b + wb)) // 2
            # 简单判定：两块 bbox 中心距离 > mask 对角线的 30% → 是分离的
            cx1, cy1 = x1a + wa // 2, y1a + ha // 2
            cx2, cy2 = x1b + wb // 2, y1b + hb // 2
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            diag = (mask.shape[1] ** 2 + mask.shape[0] ** 2) ** 0.5
            if dist > 0.25 * diag:
                return True

    return False


# ══════════════════════════════════════════════
# 三种处理模式
# ══════════════════════════════════════════════

def do_cutout(in_dir, out_dir, **kw):
    """One pipeline for root files and nested batches; originals are never modified."""
    source, target = Path(in_dir), Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    method = kw.get("method", "threshold")
    ai_method = method in {"bag", "product", "rembg"}
    model = kw.get("model", "isnet-general-use" if method == "rembg" else "birefnet-general-lite")
    size = int(kw.get("size", 1200))
    max_bytes = int(float(kw.get("max_kb", 3072)) * 1024)
    pad = int(kw.get("pad", 30))
    erode = kw.get("erode", 0 if ai_method or method == "board" else 2)
    feather = kw.get("feather", 0 if ai_method or method == "board" else 1.5)
    if "api_key" in kw:
        globals()["_removebg_key"] = kw["api_key"]
    log(f"method={method} model={model if ai_method else '-'} 输出={size}x{size}")
    records = []
    ok = err = 0
    files = sorted(p for p in source.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXT
                   and "已完成" not in p.relative_to(source).parts
                   and not p.is_relative_to(target))
    # Do not overwrite different extensions sharing the same output stem.
    outputs = {}
    for p in files:
        key = str(p.relative_to(source).with_suffix(".jpg")).casefold()
        outputs[key] = outputs.get(key, 0) + 1
    for i, p in enumerate(files, 1):
        relative = p.relative_to(source)
        record = {"source": str(relative), "status": "error", "warnings": []}
        print(f"-- {i}/{len(files)} --")
        try:
            if outputs[str(relative.with_suffix(".jpg")).casefold()] > 1:
                raise ValueError("同目录存在同名不同后缀图片，请改名以免输出互相覆盖")
            with Image.open(p) as opened:
                img = rotate(opened).convert("RGB")
            if method == "board":
                img.thumbnail((max(2400, size),) * 2, Image.Resampling.LANCZOS)
                mask, bbox, report = segment_board(img)
                img, mask, report = orient_board(img, mask, report)
                record.update(report)
            elif ai_method:
                # Bound memory for 40+ MP phone images; output still uses up to 2400px detail.
                img.thumbnail((max(2400, size),) * 2, Image.Resampling.LANCZOS)
                mask, bbox, report = segment_product(
                    img, model=model, fill_holes=kw.get("fill_holes", method != "rembg"),
                    refine=kw.get("refine_edges", True))
                record.update(report)
            else:
                mask, bbox = find_subject(img, kw.get("black_th", BLACK_TH), method=method)
            if mask is None:
                raise ValueError("未找到主体")
            sq = compose_square(img, mask, size=size, pad=pad, erode=erode, feather=feather)
            data = encode_square(sq, max_bytes, q_min=int(kw.get("q_min", Q_MIN)))
            destination = target / relative.with_suffix(".jpg")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            if kw.get("save_mask", False):
                Image.fromarray(mask).save(destination.with_name(destination.stem + "_mask.png"))
            record.update(status="review" if record["warnings"] else "ok",
                          output=str(destination.relative_to(target)), size=size, bytes=len(data))
            for warning in record["warnings"]:
                log(f"{p.name}: {warning}", "warn")
            log(f"{p.name}: {size}x{size}, {len(data)/1024:.0f}KB", "ok")
            ok += 1
        except Exception as e:
            record["error"] = str(e)
            log(f"{p.name}: {e}", "err")
            err += 1
        records.append(record)
    if kw.get("save_report", False):
        (target / "处理报告.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    review_count = sum(r["status"] == "review" for r in records)
    if review_count:
        log(f"{review_count} 张需复核，请查看上方对应文件名的提示", "warn")
    return ok, err


def do_compress(in_dir, out_dir):
    subs = subdirs(in_dir)
    if subs:
        total = 0
        for sub in subs:
            sub_out = Path(out_dir) / sub.name
            sub_out.mkdir(parents=True, exist_ok=True)
            print(f"\n  [{sub.name}]")
            files = imgs(sub)
            for i, p in enumerate(files, 1):
                print(f"-- {i}/{len(files)} --")
                try:
                    log(p.name)
                    img = rotate(Image.open(p)).convert("RGB")
                    data = compress(img, COMPRESS_MB*1024*1024, Q_INIT, Q_MIN, COMPRESS_W)
                    (sub_out / f"{p.stem}.jpg").write_bytes(data)
                    log(f"{img.size} -> {len(data)/1024:.0f}KB", "ok"); total += 1
                except Exception as e:
                    log(str(e), "err")
                print()
        return total

    files = imgs(in_dir)
    if not files: return 0
    for i, p in enumerate(files, 1):
        print(f"-- {i}/{len(files)} --")
        try:
            log(p.name)
            img = rotate(Image.open(p)).convert("RGB")
            data = compress(img, COMPRESS_MB*1024*1024, Q_INIT, Q_MIN, COMPRESS_W)
            (Path(out_dir)/f"{p.stem}.jpg").write_bytes(data)
            log(f"{img.size} -> {len(data)/1024:.0f}KB", "ok")
        except Exception as e:
            log(str(e), "err")
        print()
    return len(files)


def do_y1c(in_dir, out_dir, size=800, max_kb=2000):
    subs = subdirs(in_dir)
    if subs:
        total = 0
        for sub in subs:
            sub_out = Path(out_dir) / sub.name
            sub_out.mkdir(parents=True, exist_ok=True)
            print(f"\n  [{sub.name}]")
            files = imgs(sub)
            for i, p in enumerate(files, 1):
                print(f"-- {i}/{len(files)} --")
                try:
                    log(p.name)
                    img = rotate(Image.open(p)).convert("RGB")
                    ratio = size/max(img.size)
                    img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
                    canvas = Image.new("RGB", (size, size), (255,255,255))
                    canvas.paste(img, ((size-img.width)//2, (size-img.height)//2))
                    data = compress(canvas, max_kb*1024, Q_INIT, Q_MIN, min_w=size)
                    (sub_out / f"{p.stem}.jpg").write_bytes(data)
                    log(f"{img.size} -> {size}x{size} {len(data)/1024:.0f}KB", "ok"); total += 1
                except Exception as e:
                    log(str(e), "err")
                print()
        return total

    files = imgs(in_dir)
    if not files: return 0
    for i, p in enumerate(files, 1):
        print(f"-- {i}/{len(files)} --")
        try:
            log(p.name)
            img = rotate(Image.open(p)).convert("RGB")
            ratio = size/max(img.size)
            img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
            canvas = Image.new("RGB", (size, size), (255,255,255))
            canvas.paste(img, ((size-img.width)//2, (size-img.height)//2))
            data = compress(canvas, max_kb*1024, Q_INIT, Q_MIN, min_w=size)
            (Path(out_dir)/f"{p.stem}.jpg").write_bytes(data)
            log(f"{img.size} -> {size}x{size} {len(data)/1024:.0f}KB", "ok")
        except Exception as e:
            log(str(e), "err")
        print()
    return len(files)


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

def main():
    print("="*40)
    print("  图片批量处理 v8")
    print("="*40)

    cfg = load_config()

    total_ok = total_err = total_in = 0
    known = {"black_th","erode","pad","feather","method","api_key","size","max_kb","model","fill_holes","refine_edges","save_mask","save_report","q_min"}

    # 先统计输入图片数
    for mode_name, mode_cfg in cfg.items():
        for entry in mode_cfg.get("folders",[]):
            name = entry["name"] if isinstance(entry, dict) else entry
            in_d = BASE/name
            if mode_name == "抠图模式":
                total_in += sum(1 for p in in_d.rglob("*") if p.is_file()
                                and p.suffix.lower() in EXT
                                and "已完成" not in p.relative_to(in_d).parts)
            else:
                subs = subdirs(in_d)
                if subs:
                    for sub in subs: total_in += len(imgs(sub))
                else:
                    total_in += len(imgs(in_d))

    for mode_name, mode_cfg in cfg.items():
        desc = mode_cfg.get("description","")
        for entry in mode_cfg.get("folders",[]):
            if isinstance(entry, str):
                name, overrides = entry, {}
            else:
                name = entry["name"]
                overrides = {k:v for k,v in entry.items() if k!="name" and k in known}

            in_d  = BASE/name
            out_d = in_d/"已完成"
            out_d.mkdir(parents=True, exist_ok=True)
            if mode_name != "抠图模式" and not imgs(in_d) and not subdirs(in_d): continue

            print(f"\n[{name}] ({desc})")

            if mode_name == "抠图模式":
                ok, err = do_cutout(in_d, out_d, **overrides)
                print(f"  成功{ok} 失败{err}")
                total_ok+=ok; total_err+=err
            elif mode_name == "压缩模式":
                n = do_compress(in_d, out_d)
                print(f"  成功{n}")
                total_ok+=n
            elif mode_name == "1药城模式":
                n = do_y1c(in_d, out_d, size=overrides.get("size", 800), max_kb=overrides.get("max_kb", 2000))
                print(f"  成功{n}")
                total_ok+=n

    print(f"\n  全部完成。成功{total_ok} 失败{total_err}")
    print("="*40)

    # 处理结束弹窗提醒抽查
    try:
        import tkinter as tk
        root = tk.Tk()
        root.title("处理完成")
        root.geometry("340x180")
        root.resizable(False, False)
        root.configure(bg="white")
        root.eval(f'tk::PlaceWindow {root._w} center')

        tk.Label(root, text="处理完成", font=("Microsoft YaHei UI", 14, "bold"),
                 fg="#1a1a1a", bg="white").pack(pady=(24, 4))
        tk.Label(root, text=f"共输入 {total_in} 张，完成 {total_ok} 张",
                 font=("Microsoft YaHei UI", 12), fg="#555555", bg="white").pack()
        tk.Label(root, text="请检查输出图片质量",
                 font=("Microsoft YaHei UI", 10), fg="#999999", bg="white").pack(pady=(4, 14))

        btn = tk.Label(root, text="完  成", font=("Microsoft YaHei UI", 12),
                       fg="white", bg="#4CAF50", padx=32, pady=6,
                       cursor="hand2")
        btn.pack()
        btn.bind("<Button-1>", lambda e: [root.destroy(), exit(0)])

        root.focus_force()
        root.mainloop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
