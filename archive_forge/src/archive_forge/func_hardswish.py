from typing import List, Optional
import warnings
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2
from .modules.utils import _pair_from_first
def hardswish(input: Tensor, scale: float, zero_point: int) -> Tensor:
    """This is the quantized version of :func:`~torch.nn.functional.hardswish`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardswish' must be quantized!")
    return torch._ops.ops.quantized.hardswish(input, scale, zero_point)