from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
@_register_kernel_internal(get_num_channels, tv_tensors.Video, tv_tensor_wrapper=False)
def get_num_channels_video(video: torch.Tensor) -> int:
    return get_num_channels_image(video)