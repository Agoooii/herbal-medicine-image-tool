"""
图片批量处理 v7

用法：图片命名好 → 按类型放进对应文件夹 → 双击运行 → 取结果

四种检测方法（在 folders.ini 里配）：
  threshold — 灰度阈值（标尺图用）
  color     — 三通道色彩判断（浅色袋装图可用）
  adaptive  — 多线索融合 HSV+RGB+边缘约束（袋装图推荐，解决深色饮片/透明边缘）
  rembg     — AI背景移除（需先下载模型，GitHub被墙时不可用）

Debug 模式：设置 DEBUG=True 可保存每一步的中间结果到 调试/ 目录

依赖：pip install opencv-python pillow numpy
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
from PIL import Image, ExifTags
from datetime import datetime

BASE = Path(__file__).parent
CONFIG_FILE = BASE / "folders.ini"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def log(msg, level="info"):
    tag = {"info": "  ", "ok": "OK ", "warn": "!! ", "err": "XX "}[level]
    print(f"[{datetime.now():%H:%M:%S}] {tag}{msg}")


def save_debug(img_arr, name, prefix=""):
    """保存调试用的中间结果图片到 调试/ 目录。img_arr 为 BGR numpy 数组或 None。"""
    if not DEBUG or img_arr is None:
        return
    debug_dir = BASE / "调试"
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
            {"name": "药师帮_标尺图",     "method": "threshold", "black_th": 45, "erode": 2, "pad": 30, "feather": 1.5},
            {"name": "药师帮_袋装产品图", "method": "rembg",    "erode": 0, "pad": 30, "feather": 0},
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
    try:
        exif = img._getexif()
        if exif:
            for tid, val in exif.items():
                if ExifTags.TAGS.get(tid, tid) == "Orientation":
                    if val == 3: return img.rotate(180, expand=True)
                    if val == 6: return img.rotate(-90, expand=True)
                    if val == 8: return img.rotate(90, expand=True)
    except: pass
    return img


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

def find_subject(img, th=45, method="threshold"):
    """
    threshold: 灰度阈值 — 像素灰度>阈值=主体（标尺图）
    color:     三通道分别判断 — R>th 或 G>th 或 B>th = 主体（袋装图）
    adaptive:  多线索融合 — HSV饱和度+RGB+边缘约束（袋装图推荐）
    rembg:     AI模型（需预先下载 u2net.onnx 到 ~/.u2net/）
    """
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

    # ---- rembg (isnet模型) ----
    if method == "rembg":
        try:
            from rembg import remove, new_session
        except ImportError:
            log("rembg未安装: pip install rembg", "err")
            return None, None

        session = new_session("isnet-general-use")

        REMBG_MAX = 2000
        orig_w, orig_h = img.size
        if max(orig_w, orig_h) > REMBG_MAX:
            scale = REMBG_MAX / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            scale = 1.0
            img_small = img

        buf = io.BytesIO()
        img_small.save(buf, format="PNG")
        try:
            result = remove(buf.getvalue(), session=session)
        except Exception as e:
            log(f"rembg失败: {e}", "err")
            return None, None

        rgba = Image.open(io.BytesIO(result)).convert("RGBA")
        alpha_small = np.array(rgba.split()[3])

        if scale < 1.0:
            alpha_img = Image.fromarray(alpha_small)
            alpha_img = alpha_img.resize((orig_w, orig_h), Image.LANCZOS)
            alpha = np.array(alpha_img)
        else:
            alpha = alpha_small

        fg = alpha > 10
        if not fg.any():
            return None, None
        rows, cols = np.where(fg)
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        if (x2-x1)*(y2-y1) < 0.03 * w * h:
            return None, None
        return alpha, (x1, y1, x2-x1, y2-y1)

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
        threshold = max(bg_mad * 3 + 30, 35)
        threshold = min(threshold, 80)

        if DEBUG:
            log(f"adaptive: bg_MAD={bg_mad:.1f} threshold={threshold:.0f}")

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
            fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
            fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

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
    nw = ~np.all(arr>248, axis=2)
    if nw.any():
        rs, cs = np.any(nw, axis=1), np.any(nw, axis=0)
        y1, y2 = int(np.where(rs)[0][0]), int(np.where(rs)[0][-1])
        x1, x2 = int(np.where(cs)[0][0]), int(np.where(cs)[0][-1])
    else:
        y1, y2, x1, x2 = 0, img.height-1, 0, img.width-1
    x1, y1 = max(0,x1-pad), max(0,y1-pad)
    x2, y2 = min(img.width-1, x2+pad), min(img.height-1, y2+pad)
    tight = img.crop((x1, y1, x2+1, y2+1))
    side = max(tight.size)
    sq = Image.new("RGB", (side,side), (255,255,255))
    sq.paste(tight, ((side-tight.width)//2, (side-tight.height)//2))
    return sq


def needs_review(sq_img=None, mask=None, orig_w=1, orig_h=1, min_product_pct=20, min_mask_pct=25):
    """检查产品是否残缺
    - 输出图非白占比 < min_product_pct → 大面积丢失
    - 原图蒙版覆盖率 < min_mask_pct → 产品被部分吃掉
    """
    if sq_img is not None:
        arr = np.array(sq_img)
        non_white = np.sum(np.max(arr, axis=2) < 240)
        pct = non_white / (sq_img.width * sq_img.height) * 100
        if pct < min_product_pct:
            return True
    if mask is not None:
        mask_pct = np.sum(mask > 0) / (orig_w * orig_h) * 100
        if mask_pct < min_mask_pct:
            return True
    return False


# ══════════════════════════════════════════════
# 三种处理模式
# ══════════════════════════════════════════════

def do_cutout(in_dir, out_dir, **kw):
    th = kw.get("black_th", BLACK_TH)
    pa = kw.get("pad", PAD)
    fe = kw.get("feather", FEATHER)
    er = kw.get("erode", ERODE)
    md = kw.get("method", "threshold")
    if "api_key" in kw:
        globals()["_removebg_key"] = kw["api_key"]
    log(f"  method={md} th={th} erode={er} feather={fe} pad={pa}")

    # 检测子文件夹：如果有，按文件夹批量处理
    subs = subdirs(in_dir)
    if subs:
        # 子文件夹模式：每个子文件夹单独输出到 已完成/子文件夹名/
        total_ok = total_err = 0
        for sub in subs:
            sub_name = sub.name
            sub_out = Path(out_dir) / sub_name
            sub_out.mkdir(parents=True, exist_ok=True)
            print(f"\n  [{sub_name}]")
            files = imgs(sub)
            ok = err = 0
            for i, p in enumerate(files, 1):
                print(f"-- {i}/{len(files)} --")
                try:
                    log(p.name)
                    img = rotate(Image.open(p).convert("RGB"))
                    cv_orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    if md == "adaptive":
                        cv_orig = apply_gamma_for_adaptive(cv_orig)
                        img = Image.fromarray(cv2.cvtColor(cv_orig, cv2.COLOR_BGR2RGB))
                    mask, bbox = find_subject(img, th, method=md)
                    if DEBUG:
                        save_debug(cv_orig, "01_original", p.stem)
                        if mask is not None:
                            save_debug(mask, "02_mask", p.stem)
                            overlay = cv_orig.copy()
                            overlay[mask == 0] = overlay[mask == 0] // 2
                            save_debug(overlay, "03_overlay", p.stem)
                    if mask is None:
                        log("未找到主体", "warn")
                        err += 1; print(); continue
                    white = white_bg(img, mask, bbox, pa, fe, er)
                    if DEBUG:
                        cv_white = cv2.cvtColor(np.array(white), cv2.COLOR_RGB2BGR)
                        save_debug(cv_white, "04_white_bg", p.stem)
                    sq = square(white, pa).resize((1200, 1200), Image.LANCZOS)
                    if DEBUG:
                        cv_sq = cv2.cvtColor(np.array(sq), cv2.COLOR_RGB2BGR)
                        save_debug(cv_sq, "05_square", p.stem)
                    data = compress(sq, CUTOUT_MB*1024*1024, Q_INIT, Q_MIN, CUTOUT_W)
                    (sub_out / f"{p.stem}.jpg").write_bytes(data)
                    log(f"{sq.size} {len(data)/1024:.0f}KB", "ok"); ok += 1
                except Exception as e:
                    log(str(e), "err")
                    err += 1
                print()
            print(f"  {sub_name}: 成功{ok} 失败{err}")
            total_ok += ok; total_err += err
        return total_ok, total_err

    # 无子文件夹：处理根目录下的图片（原逻辑）
    files = imgs(in_dir)
    if not files: return 0,0
    ok = err = 0
    for i, p in enumerate(files, 1):
        print(f"-- {i}/{len(files)} --")
        try:
            log(p.name)
            img = rotate(Image.open(p).convert("RGB"))
            cv_orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if md == "adaptive":
                cv_orig = apply_gamma_for_adaptive(cv_orig)
                img = Image.fromarray(cv2.cvtColor(cv_orig, cv2.COLOR_BGR2RGB))
            mask, bbox = find_subject(img, th, method=md)

            if DEBUG:
                save_debug(cv_orig, "01_original", p.stem)
                if mask is not None:
                    save_debug(mask, "02_mask", p.stem)
                    overlay = cv_orig.copy()
                    overlay[mask == 0] = overlay[mask == 0] // 2
                    save_debug(overlay, "03_overlay", p.stem)

            if mask is None:
                log("未找到主体", "warn")
                err += 1; print(); continue
            white = white_bg(img, mask, bbox, pa, fe, er)
            if DEBUG:
                cv_white = cv2.cvtColor(np.array(white), cv2.COLOR_RGB2BGR)
                save_debug(cv_white, "04_white_bg", p.stem)
            sq = square(white, pa).resize((1200, 1200), Image.LANCZOS)
            if DEBUG:
                cv_sq = cv2.cvtColor(np.array(sq), cv2.COLOR_RGB2BGR)
                save_debug(cv_sq, "05_square", p.stem)
            data = compress(sq, CUTOUT_MB*1024*1024, Q_INIT, Q_MIN, CUTOUT_W)
            (Path(out_dir)/f"{p.stem}.jpg").write_bytes(data)
            log(f"{sq.size} {len(data)/1024:.0f}KB", "ok"); ok += 1
        except Exception as e:
            log(str(e), "err")
            err += 1
        print()
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
                    img = rotate(Image.open(p).convert("RGB"))
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
            img = rotate(Image.open(p).convert("RGB"))
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
                    img = rotate(Image.open(p).convert("RGB"))
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
            img = rotate(Image.open(p).convert("RGB"))
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
    print("  图片批量处理 v6")
    print("="*40)

    cfg = load_config()

    total_ok = total_err = total_in = 0
    known = {"black_th","erode","pad","feather","method","api_key","size","max_kb"}

    # 先统计输入图片数
    for mode_name, mode_cfg in cfg.items():
        for entry in mode_cfg.get("folders",[]):
            name = entry["name"] if isinstance(entry, dict) else entry
            in_d = BASE/name
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
            if not imgs(in_d) and not subdirs(in_d): continue

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
