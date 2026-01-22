import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def get_image_num_channels(img: Tensor) -> int:
    _assert_image_tensor(img)
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]
    raise TypeError(f'Input ndim should be 2 or more. Got {img.ndim}')