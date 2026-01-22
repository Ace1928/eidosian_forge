import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from .image_utils import (
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
def corners_to_center_format(bboxes_corners: TensorType) -> TensorType:
    """
    Converts bounding boxes from corners format to center format.

    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    center format: contains the coordinate for the center of the box and its the width, height dimensions
        (center_x, center_y, width, height)
    """
    if is_torch_tensor(bboxes_corners):
        return _corners_to_center_format_torch(bboxes_corners)
    elif isinstance(bboxes_corners, np.ndarray):
        return _corners_to_center_format_numpy(bboxes_corners)
    elif is_tf_tensor(bboxes_corners):
        return _corners_to_center_format_tf(bboxes_corners)
    raise ValueError(f'Unsupported input type {type(bboxes_corners)}')