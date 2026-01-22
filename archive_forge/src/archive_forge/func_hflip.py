import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def hflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)
    return img.flip(-1)