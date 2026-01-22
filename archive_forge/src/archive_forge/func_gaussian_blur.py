import math
from typing import List, Optional
import PIL.Image
import torch
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms._functional_tensor import _max_value
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
def gaussian_blur(inpt: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]]=None) -> torch.Tensor:
    """[BETA] See :class:`~torchvision.transforms.v2.GaussianBlur` for details."""
    if torch.jit.is_scripting():
        return gaussian_blur_image(inpt, kernel_size=kernel_size, sigma=sigma)
    _log_api_usage_once(gaussian_blur)
    kernel = _get_kernel(gaussian_blur, type(inpt))
    return kernel(inpt, kernel_size=kernel_size, sigma=sigma)