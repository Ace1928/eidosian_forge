from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, logging, requires_backends
def build_palette(num_labels: int) -> List[Tuple[int, int]]:
    base = int(num_labels ** (1 / 3)) + 1
    margin = 256 // base
    color_list = [(0, 0, 0)]
    for location in range(num_labels):
        num_seq_r = location // base ** 2
        num_seq_g = location % base ** 2 // base
        num_seq_b = location % base
        R = 255 - num_seq_r * margin
        G = 255 - num_seq_g * margin
        B = 255 - num_seq_b * margin
        color_list.append((R, G, B))
    return color_list