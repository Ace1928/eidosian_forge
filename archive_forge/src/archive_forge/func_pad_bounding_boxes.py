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
def pad_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], padding: List[int], padding_mode: str='constant') -> Tuple[torch.Tensor, Tuple[int, int]]:
    if padding_mode not in ['constant']:
        raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")
    left, right, top, bottom = _parse_pad_padding(padding)
    if format == tv_tensors.BoundingBoxFormat.XYXY:
        pad = [left, top, left, top]
    else:
        pad = [left, top, 0, 0]
    bounding_boxes = bounding_boxes + torch.tensor(pad, dtype=bounding_boxes.dtype, device=bounding_boxes.device)
    height, width = canvas_size
    height += top + bottom
    width += left + right
    canvas_size = (height, width)
    return (clamp_bounding_boxes(bounding_boxes, format=format, canvas_size=canvas_size), canvas_size)