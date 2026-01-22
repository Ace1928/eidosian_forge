import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def ones_(tensor: Tensor) -> Tensor:
    """Fill the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    return _no_grad_fill_(tensor, 1.0)