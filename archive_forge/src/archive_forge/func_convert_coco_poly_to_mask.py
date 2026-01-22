import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def convert_coco_poly_to_mask(self, *args, **kwargs):
    logger.warning_once('The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ')
    return convert_coco_poly_to_mask(*args, **kwargs)