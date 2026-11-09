import cv2
import numpy as np
from .geometry import crop_normalized
from .schema import Brick

def detect_top_bricks(image, config):
    roi, offset = crop_normalized(image, config["top_roi"])
    ox, oy, _, _ = offset
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Local contrast helps separate mortar gaps from brick faces.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, int(config["canny_low"]), int(config["canny_high"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = roi.shape[:2]
    min_area = float(config["min_brick_area_ratio"]) * rw * rh
    max_area = float(config["max_brick_area_ratio"]) * rw * rh
    bricks = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if h < 1 or w < 1:
            continue
        aspect = w / h
        if not (config["min_aspect"] <= aspect <= config["max_aspect"]):
            continue
        # Reject large outer pallet frame contours.
        if w > rw*0.8 or h > rh*0.8:
            continue
        bricks.append(Brick(x+ox, y+oy, x+w+ox, y+h+oy, 0.75))

    # Remove near-identical nested/duplicate contours.
    bricks.sort(key=lambda b: (b.y1, b.x1, -((b.x2-b.x1)*(b.y2-b.y1))))
    unique = []
    for b in bricks:
        cx, cy = (b.x1+b.x2)/2, (b.y1+b.y2)/2
        bw, bh = b.x2-b.x1, b.y2-b.y1
        duplicate = False
        for q in unique:
            qx, qy = (q.x1+q.x2)/2, (q.y1+q.y2)/2
            qw, qh = q.x2-q.x1, q.y2-q.y1
            if abs(cx-qx) < min(bw,qw)*0.25 and abs(cy-qy) < min(bh,qh)*0.25:
                duplicate = True
                break
        if not duplicate:
            unique.append(b)
    return roi, unique
