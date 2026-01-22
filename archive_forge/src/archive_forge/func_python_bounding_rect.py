from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, logging
from ...utils.import_utils import is_cv2_available, is_vision_available
def python_bounding_rect(self, coordinates):
    """This is a reimplementation of a BoundingRect function equivalent to cv2."""
    min_values = np.min(coordinates, axis=(0, 1)).astype(int)
    max_values = np.max(coordinates, axis=(0, 1)).astype(int)
    x_min, y_min = (min_values[0], min_values[1])
    width = max_values[0] - x_min + 1
    height = max_values[1] - y_min + 1
    return (x_min, y_min, width, height)