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
@_register_five_ten_crop_kernel_internal(ten_crop, torch.Tensor)
@_register_five_ten_crop_kernel_internal(ten_crop, tv_tensors.Image)
def ten_crop_image(image: torch.Tensor, size: List[int], vertical_flip: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    non_flipped = five_crop_image(image, size)
    if vertical_flip:
        image = vertical_flip_image(image)
    else:
        image = horizontal_flip_image(image)
    flipped = five_crop_image(image, size)
    return non_flipped + flipped