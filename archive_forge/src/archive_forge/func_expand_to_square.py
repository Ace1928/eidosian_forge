import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict, select_best_resolution
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def expand_to_square(image: np.array, background_color, input_data_format) -> np.array:
    """
    Expands an image to a square by adding a background color.
    """
    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image
    elif width > height:
        result = np.ones((width, width, image.shape[2]), dtype=image.dtype) * background_color
        result[(width - height) // 2:(width - height) // 2 + height, :] = image
        return result
    else:
        result = np.ones((height, height, image.shape[2]), dtype=image.dtype) * background_color
        result[:, (height - width) // 2:(height - width) // 2 + width] = image
        return result