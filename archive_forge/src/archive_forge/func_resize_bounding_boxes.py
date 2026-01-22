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
def resize_bounding_boxes(bounding_boxes: torch.Tensor, canvas_size: Tuple[int, int], size: List[int], max_size: Optional[int]=None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    old_height, old_width = canvas_size
    new_height, new_width = _compute_resized_output_size(canvas_size, size=size, max_size=max_size)
    if (new_height, new_width) == (old_height, old_width):
        return (bounding_boxes, canvas_size)
    w_ratio = new_width / old_width
    h_ratio = new_height / old_height
    ratios = torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=bounding_boxes.device)
    return (bounding_boxes.mul(ratios).to(bounding_boxes.dtype), (new_height, new_width))