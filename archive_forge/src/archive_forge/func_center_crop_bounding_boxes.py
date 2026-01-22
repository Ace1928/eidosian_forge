import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def center_crop_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], output_size: List[int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, *canvas_size)
    return crop_bounding_boxes(bounding_boxes, format, top=crop_top, left=crop_left, height=crop_height, width=crop_width)