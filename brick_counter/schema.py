from dataclasses import dataclass
import numpy as np

@dataclass
class Brick:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

@dataclass
class CountResult:
    top_view: np.ndarray
    side_view: np.ndarray
    bricks: list[Brick]
    seams: list[int]
    bricks_per_layer: int
    layers: int
    total: int
    top_count_source: str
    layer_count_source: str
