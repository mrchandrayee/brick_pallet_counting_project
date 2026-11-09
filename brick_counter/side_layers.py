import cv2
import numpy as np
from .geometry import crop_normalized

def detect_layer_seams(image, config):
    roi, _ = crop_normalized(image, config["side_roi"])
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 35, 120)
    # Horizontal edge energy, averaged over image width.
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, roi.shape[1]//18), 1)))
    projection = horizontal.sum(axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(projection.reshape(-1,1), (1, nine_odd(config["projection_smooth"])), 0).ravel()
    threshold = max(float(config["seam_min_strength"]), float(smooth.max()) * config["seam_relative_strength"])
    candidates = np.where(smooth >= threshold)[0].tolist()
    # Collapse adjacent rows into one seam location.
    groups = []
    for y in candidates:
        if not groups or y - groups[-1][-1] > int(config["seam_merge_distance"]):
            groups.append([y])
        else:
            groups[-1].append(y)
    peaks = [int(round(sum(g)/len(g))) for g in groups]
    # Ignore ROI boundaries; layers are intervals between internal seams.
    peaks = [y for y in peaks if 4 < y < roi.shape[0]-4]
    min_gap = int(config["min_layer_height"])
    filtered = []
    for y in peaks:
        if not filtered or y-filtered[-1] >= min_gap:
            filtered.append(y)
        elif smooth[y] > smooth[filtered[-1]]:
            filtered[-1] = y
    return roi, filtered

def nine_odd(v):
    v = max(1, int(v))
    return v if v % 2 else v+1
