import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(erase, torch.Tensor)
@_register_kernel_internal(erase, tv_tensors.Image)
def erase_image(image: torch.Tensor, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    if not inplace:
        image = image.clone()
    image[..., i:i + h, j:j + w] = v
    return image