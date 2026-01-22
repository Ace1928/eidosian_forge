from typing import List
import PIL.Image
import torch
from torch.nn.functional import conv2d
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value
from torchvision.utils import _log_api_usage_once
from ._misc import _num_value_bits, to_dtype_image
from ._type_conversion import pil_to_tensor, to_pil_image
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(solarize, torch.Tensor)
@_register_kernel_internal(solarize, tv_tensors.Image)
def solarize_image(image: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold > _max_value(image.dtype):
        raise TypeError(f'Threshold should be less or equal the maximum value of the dtype, but got {threshold}')
    return torch.where(image >= threshold, invert_image(image), image)