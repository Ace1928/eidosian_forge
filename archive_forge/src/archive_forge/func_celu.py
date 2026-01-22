from typing import List, Optional
import warnings
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2
from .modules.utils import _pair_from_first
def celu(input: Tensor, scale: float, zero_point: int, alpha: float=1.0) -> Tensor:
    """celu(input, scale, zero_point, alpha=1.) -> Tensor

    Applies the quantized CELU function element-wise.

    .. math::
        \\text{CELU}(x) = \\max(0,x) + \\min(0, \\alpha * (\\exp(x / \\alpha) - 1))

    Args:
        input: quantized input
        alpha: the :math:`\\alpha` value for the CELU formulation. Default: 1.0
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.celu' must be quantized!")
    return torch.ops.quantized.celu(input, scale, zero_point, alpha)