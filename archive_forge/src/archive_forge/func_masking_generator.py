import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
@lru_cache()
def masking_generator(self, input_size_patches, total_mask_patches, mask_group_min_patches, mask_group_max_patches, mask_group_min_aspect_ratio, mask_group_max_aspect_ratio) -> FlavaMaskingGenerator:
    return FlavaMaskingGenerator(input_size=input_size_patches, total_mask_patches=total_mask_patches, mask_group_min_patches=mask_group_min_patches, mask_group_max_patches=mask_group_max_patches, mask_group_min_aspect_ratio=mask_group_min_aspect_ratio, mask_group_max_aspect_ratio=mask_group_max_aspect_ratio)