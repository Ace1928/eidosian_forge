import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def scale_factor_to_fit(original_size, target_size=None):
    height, width = original_size
    if target_size is None:
        max_height = self.image_processor.size['height']
        max_width = self.image_processor.size['width']
    else:
        max_height, max_width = target_size
    if width <= max_width and height <= max_height:
        return 1.0
    return min(max_height / height, max_width / width)