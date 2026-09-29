"""Whole ruler-board segmentation; preserve all content inside the board perimeter.

The mask follows the board rather than the brightness/color of herbs or grid cells.
The boundary mask retains perspective; orient_board rotates the full board upright.
"""
import cv2
import numpy as np
from PIL import Image
from product_cutout import mask_bbox


def _side_evidence(lab, quad):
    h, w = lab.shape[:2]
    center = quad.mean(axis=0)
    chroma, light = [], []
    for i in range(4):
        t = np.linspace(.12, .88, 60)[:, None]
        points = quad[i] * (1-t) + quad[(i+1) % 4] * t
        normal = center - (quad[i] + quad[(i+1) % 4]) / 2
        normal /= max(np.linalg.norm(normal), 1e-6)
        distance = max(3, min(h, w) * .008)
        inside = np.clip(np.round(points + normal * distance).astype(int), [0, 0], [w-1, h-1])
        outside = np.clip(np.round(points - normal * distance).astype(int), [0, 0], [w-1, h-1])
        a = lab[inside[:, 1], inside[:, 0]].astype(float)
        b = lab[outside[:, 1], outside[:, 0]].astype(float)
        chroma.append(float(np.median(np.linalg.norm(a[:, 1:] - b[:, 1:], axis=1))))
        light.append(float(np.median(abs(a[:, 0] - b[:, 0]))))
    return np.array(chroma), np.array(light)


def _board_candidates(rgb):
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    seeds = []
    # Several thresholds propose boundaries; none is used to classify the board interior.
    # Color seeds help with the cool-tinted laminated grid board on reflective cream tables.
    for threshold in (112, 116, 120, 124, 128, 132):
        seeds.append(('color', (lab[:, :, 2] < threshold).astype(np.uint8) * 255))
    color_delta = lab[:, :, 1].astype(float) - lab[:, :, 2]
    for threshold in (0, 4, 8, 12):
        seeds.append(('color', (color_delta > threshold).astype(np.uint8) * 255))
    # Luminance edges also support neutral boards and non-black backgrounds.
    blurred = cv2.GaussianBlur(gray, (3, 3), .6)
    for low, high in ((10, 30), (25, 70), (50, 140)):
        seeds.append(('edge', cv2.Canny(blurred, low, high)))
    candidates = []
    for source, seed in seeds:
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(seed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = cv2.contourArea(contour)
            if not .12 * h * w < area < .94 * h * w:
                continue
            hull = cv2.convexHull(contour)
            quad = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * .018, True).reshape(-1, 2)
            if len(quad) != 4:
                continue
            if np.any((quad < 3) | (quad > np.array([w-4, h-4])), axis=1).sum() > 1:
                continue  # The image frame is not the board.
            quad_area = cv2.contourArea(quad)
            if quad_area <= 0 or area / quad_area < .8:
                continue
            sides = np.roll(quad, -1, axis=0) - quad
            lengths = np.linalg.norm(sides, axis=1)
            if lengths.min() / max(lengths.max(), 1) < .3:
                continue
            cosine = np.sum(sides * np.roll(sides, 1, axis=0), axis=1) / (lengths * np.roll(lengths, 1))
            if np.max(np.abs(cosine)) > .55:
                continue
            chroma, light = _side_evidence(lab, quad.astype(float))
            if np.min(chroma + light * .15) < 1.5:
                continue
            score = np.sqrt(quad_area / (h*w)) * (min(chroma.mean(), 20) + min(light.mean(), 60) * .08) * min(area / quad_area, 1)
            candidates.append((float(score), quad.astype(float), contour, source))
    return sorted(candidates, key=lambda c: c[0], reverse=True), lab


def _fit_board_sides(quad, contour):
    """Fit the straight parts, so rounded corner vertices don't crop the outer white rim."""
    points = contour.reshape(-1, 2).astype(float)
    lines = []
    for i in range(4):
        start, end = quad[i], quad[(i+1) % 4]
        vector = end-start
        length = np.linalg.norm(vector)
        tangent = vector / length
        normal = np.array([-tangent[1], tangent[0]])
        along = (points-start) @ tangent
        across = (points-start) @ normal
        selected = points[(along > .12*length) & (along < .88*length) & (np.abs(across) < max(8, .02*length))]
        if len(selected) < 16:
            return quad
        vx, vy, x, y = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).flatten()
        n = np.array([-vy, vx], dtype=float)
        lines.append((n, float(n @ [x, y])))
    refined = []
    for i in range(4):
        n1, d1 = lines[i-1]; n2, d2 = lines[i]
        matrix = np.stack((n1, n2))
        if abs(np.linalg.det(matrix)) < .3:
            return quad
        refined.append(np.linalg.solve(matrix, [d1, d2]))
    refined = np.array(refined)
    if np.max(np.linalg.norm(refined-quad, axis=1)) > .03*np.linalg.norm(quad.max(axis=0)-quad.min(axis=0)):
        return quad
    return refined


def segment_board(img):
    work = img.convert('RGB').copy()
    work.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    candidates, lab = _board_candidates(np.asarray(work))
    if not candidates or candidates[0][0] < .8:
        raise ValueError('未找到完整标尺板外边界：请检查板子四角是否拍全、板边是否清晰；不会只抠取饮片替代整板')
    score, quad, contour, source = candidates[0]
    quad = _fit_board_sides(quad, contour)
    h, w = lab.shape[:2]
    chroma, light = _side_evidence(lab, quad)
    warnings = []
    if np.min(chroma + light*.15) < 3:
        warnings.append('部分板边对比度较低，请复核外沿是否完整')
    if np.any((quad < 3) | (quad > [w-4, h-4])):
        warnings.append('标尺板靠近原图边界，请检查是否已截断板角')
    # Sharp straight edges with a subpixel antialias; board interior is completely opaque.
    scale = np.array([img.width / work.width, img.height / work.height])
    full_quad = quad * scale
    mask = np.zeros((img.height, img.width), np.uint8)
    cv2.fillConvexPoly(mask, np.round(full_quad * 256).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=8)
    # Preserve observed rounded corners, while straight edges come from robust fitted lines.
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float) * scale
    corner_support = np.zeros_like(mask)
    cv2.fillConvexPoly(corner_support, np.round(hull * 256).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=8)
    support_radius = max(1, round(max(scale)))
    corner_support = cv2.dilate(corner_support, np.ones((support_radius*2+1,)*2, np.uint8))
    corner_radius = max(3, round(np.min(np.linalg.norm(np.roll(full_quad,-1,axis=0)-full_quad,axis=1))*.025))
    for x, y in np.round(full_quad).astype(int):
        left, top = max(0,x-corner_radius), max(0,y-corner_radius)
        right, bottom = min(img.width,x+corner_radius+1), min(img.height,y+corner_radius+1)
        if left < right and top < bottom:
            mask[top:bottom,left:right] = np.minimum(mask[top:bottom,left:right], corner_support[top:bottom,left:right])
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError('标尺板蒙版为空')
    report = {'detector': 'board_boundary', 'boundary_source': source,
              'board_corners': np.round(full_quad, 2).tolist(),
              'coverage': round(float(np.mean(mask >= 127)), 4), 'warnings': warnings}
    return mask, bbox, report


def orient_board(img, mask, report):
    """Find the printed multicolor scale, rotate it to the left, and remove roll.

    Perspective warping is used only for measuring four edge strips. Output pixels
    undergo rotation only, so ruler proportions and the complete board are retained.
    Ambiguous or obscured color scales fail explicitly instead of guessing upright.
    """
    quad = np.asarray(report['board_corners'], np.float32)
    preview = img.convert('RGB').copy()
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    scaled_quad = quad * np.array([preview.width/img.width, preview.height/img.height], np.float32)
    target = np.array([[0,0],[599,0],[599,599],[0,599]],np.float32)
    transform = cv2.getPerspectiveTransform(scaled_quad, target)
    view = cv2.warpPerspective(np.asarray(preview), transform, (600,600))
    hsv = cv2.cvtColor(view,cv2.COLOR_RGB2HSV)
    bands = [hsv[5:65,25:575], hsv[25:575,535:595], hsv[535:595,25:575], hsv[25:575,5:65]]
    scores = []
    group_counts = []
    for band in bands:
        hue, saturation, value = band[:,:,0], band[:,:,1], band[:,:,2]
        vivid = (saturation > 85) & (value > 65)
        groups = [vivid & ((hue < 10) | (hue > 170)),
                  vivid & (hue > 35) & (hue < 90),
                  vivid & (hue >= 90) & (hue < 140)]
        fractions = np.array([group.mean() for group in groups])
        scores.append(float(fractions.sum() + 2*fractions.min()))
        group_counts.append(int(np.sum(fractions > .001)))
    order = np.argsort(scores)
    side, runner_up = int(order[-1]), int(order[-2])
    if (scores[side] < .025 or scores[side] < scores[runner_up]*1.8
            or scores[side]-scores[runner_up] < .015 or group_counts[side] < 3):
        raise ValueError('无法可靠判断标尺板朝向：彩色色卡不完整、被遮挡或不清晰。请检查原图；本张不生成猜测朝向的成品')
    edge = quad[(side+1)%4] - quad[side]
    normal = np.array([-edge[1], edge[0]])
    midpoint = (quad[side]+quad[(side+1)%4])/2
    if np.dot(normal, quad.mean(axis=0)-midpoint) < 0:
        normal = -normal
    # Align the average of the two long edges; keep perspective without stretching.
    horizontal = ((quad[(side+2)%4]-quad[(side+1)%4])
                  + (quad[(side+3)%4]-quad[side]))
    if np.dot(horizontal, normal) < 0:
        horizontal = -horizontal
    angle = float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))
    nearest_quarter = round(angle/90)*90
    if abs(angle-nearest_quarter) < .05:
        angle = float(nearest_quarter)
    rgba = img.convert('RGBA')
    rgba.putalpha(Image.fromarray(mask))
    # Premultiplied rotation prevents dark fringes at transparent board edges.
    turned = rgba.convert('RGBa').rotate(angle, resample=Image.Resampling.BICUBIC,
                                         expand=True, fillcolor=(0,0,0,0)).convert('RGBA')
    turned_mask = np.asarray(turned.getchannel('A')).copy()
    new_report = dict(report)
    new_report['board_corners_input'] = new_report.pop('board_corners')
    new_report.update(orientation='upright_color_chart_left',
                      rotation_degrees_ccw=round(angle,3),
                      orientation_edge_scores=[round(s,4) for s in scores],
                      output_canvas_size=list(turned.size))
    return turned.convert('RGB'), turned_mask, new_report
