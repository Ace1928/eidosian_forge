from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def swiglu_packed(x: torch.Tensor, w1w2: torch.Tensor, b1b2: Optional[torch.Tensor], w3: torch.Tensor, b3: Optional[torch.Tensor], *, op: SwiGLUOp) -> torch.Tensor:
    """
    Computes a SwiGLU block given the weights/bias of the 3
    linear layers.

    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, w3, b3)

    :Supported hardware:

    This operator is only optimized on A100+ on ``torch.half`` or ``torch.bfloat16``         (autocast is supported), and will fallback to a functional pytorch         implementation otherwise.
    """
    batch_shape = x.shape[:-1]
    x = x.reshape([-1, x.shape[-1]])
    if b3 is not None:
        if b3.ndim != 1 or b3.shape[0] != w3.shape[0]:
            raise ValueError(f'Invalid shapes for w3: {w3.shape} / b3: {b3.shape}')
    assert op.PACKED_WEIGHTS, 'Not implemented PACKED_WEIGHTS'
    return op(x, w1w2, b1b2, w3, b3).reshape([*batch_shape, -1])