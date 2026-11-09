import json
from pathlib import Path
import cv2
from .top_detector import detect_top_bricks
from .side_layers import detect_layer_seams
from .schema import CountResult

class BrickPalletCounter:
    def __init__(self, config_path):
        path = Path(config_path)
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def process(self, top, side, bricks_override=None, layers_override=None):
        top_roi, bricks = detect_top_bricks(top, self.config)
        side_roi, seams = detect_layer_seams(side, self.config)
        # A visual detector is an estimate; manual overrides are provided for
        # calibrated facilities where a verified per-layer count is known.
        bpl = int(bricks_override) if bricks_override is not None else len(bricks)
        layers = int(layers_override) if layers_override is not None else max(1, len(seams)+1)
        top_source = "manual override" if bricks_override is not None else "contour estimate"
        layer_source = "manual override" if layers_override is not None else "horizontal seam estimate"
        return CountResult(top.copy(), side.copy(), bricks, seams, bpl, layers,
                           bpl*layers, top_source, layer_source)
