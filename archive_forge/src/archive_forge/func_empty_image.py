from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def empty_image(size, input_data_format):
    if input_data_format == ChannelDimension.FIRST:
        return np.zeros((3, *size), dtype=np.uint8)
    elif input_data_format == ChannelDimension.LAST:
        return np.zeros((*size, 3), dtype=np.uint8)
    raise ValueError('Invalid channel dimension format.')