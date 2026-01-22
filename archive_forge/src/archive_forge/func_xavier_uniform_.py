import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def xavier_uniform_(tensor: Tensor, gain: float=1.0, generator: _Optional[torch.Generator]=None) -> Tensor:
    """Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\\mathcal{U}(-a, a)` where

    .. math::
        a = \\text{gain} \\times \\sqrt{\\frac{6}{\\text{fan\\_in} + \\text{fan\\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -a, a, generator)