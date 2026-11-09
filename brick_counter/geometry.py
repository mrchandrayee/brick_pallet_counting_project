import cv2
import numpy as np

def crop_normalized(image, roi):
    """ROI is [x1,y1,x2,y2] in normalized 0..1 coordinates."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi
    xa, ya = int(x1*w), int(y1*h)
    xb, yb = int(x2*w), int(y2*h)
    xa, ya = max(0,xa), max(0,ya)
    xb, yb = min(w,xb), min(h,yb)
    if xb <= xa or yb <= ya:
        raise ValueError(f"Invalid normalized ROI: {roi}")
    return image[ya:yb, xa:xb].copy(), (xa,ya,xb,yb)

def perspective_warp(image, points, width=1000, height=700):
    """Optional four-point rectification; points are normalized image coordinates."""
    pts = np.asarray([[x*image.shape[1], y*image.shape[0]] for x,y in points],
                     dtype=np.float32)
    dst = np.asarray([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],
                     dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (width,height))
