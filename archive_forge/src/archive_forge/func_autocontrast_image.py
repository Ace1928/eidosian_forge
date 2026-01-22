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
@_register_kernel_internal(autocontrast, torch.Tensor)
@_register_kernel_internal(autocontrast, tv_tensors.Image)
def autocontrast_image(image: torch.Tensor) -> torch.Tensor:
    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f'Input image tensor permitted channel values are 1 or 3, but found {c}')
    if image.numel() == 0:
        return image
    bound = _max_value(image.dtype)
    fp = image.is_floating_point()
    float_image = image if fp else image.to(torch.float32)
    minimum = float_image.amin(dim=(-2, -1), keepdim=True)
    maximum = float_image.amax(dim=(-2, -1), keepdim=True)
    eq_idxs = maximum == minimum
    inv_scale = maximum.sub_(minimum).mul_(1.0 / bound)
    minimum[eq_idxs] = 0.0
    inv_scale[eq_idxs] = 1.0
    if fp:
        diff = float_image.sub(minimum)
    else:
        diff = float_image.sub_(minimum)
    return diff.div_(inv_scale).clamp_(0, bound).to(image.dtype)