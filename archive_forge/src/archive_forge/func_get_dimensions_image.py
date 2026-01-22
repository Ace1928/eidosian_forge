from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
@_register_kernel_internal(get_dimensions, torch.Tensor)
@_register_kernel_internal(get_dimensions, tv_tensors.Image, tv_tensor_wrapper=False)
def get_dimensions_image(image: torch.Tensor) -> List[int]:
    chw = list(image.shape[-3:])
    ndims = len(chw)
    if ndims == 3:
        return chw
    elif ndims == 2:
        chw.insert(0, 1)
        return chw
    else:
        raise TypeError(f'Input tensor should have at least two dimensions, but got {ndims}')