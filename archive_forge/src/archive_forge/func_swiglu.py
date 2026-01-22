from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def swiglu(x: torch.Tensor, w1: torch.Tensor, b1: Optional[torch.Tensor], w2: torch.Tensor, b2: Optional[torch.Tensor], w3: torch.Tensor, b3: Optional[torch.Tensor], *, op: Optional[SwiGLUOp]=None) -> torch.Tensor:
    """
    Computes a SwiGLU block given the weights/bias of the 3
    linear layers.

    - It is recommended to keep ``op=None`` so the best implementation     available for the inputs will be used.


    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, w3, b3)

    :Packing weights:

    To allow faster implementations, it's recommended to have w1/w2 come from the same storage, as in:
        .. code-block:: python

            w1, w2 = xformers.ops.unbind(w12, 0)

    :Supported hardware:

    This operator is only optimized on A100+ on ``torch.half`` or ``torch.bfloat16``         (autocast is supported), and will fallback to a functional pytorch         implementation otherwise.
    """
    batch_shape = x.shape[:-1]
    x = x.reshape([-1, x.shape[-1]])
    if w1.ndim != 2 or w1.shape != w2.shape:
        raise ValueError(f'Invalid shapes for w1: {w1.shape} / w2: {w2.shape}')
    if b1 is not None:
        if b1.ndim != 1 or b1.shape[0] != w1.shape[0]:
            raise ValueError(f'Invalid shapes for b1: {b1.shape}')
    if b2 is not None:
        if b2.ndim != 1 or b2.shape[0] != w2.shape[0]:
            raise ValueError(f'Invalid shapes for b2: {b2.shape}')
    if w3.ndim != 2 or w3.shape[1] != w2.shape[0]:
        raise ValueError(f'Invalid shape for w3: {w3.shape}')
    if b3 is not None:
        if b3.ndim != 1 or b3.shape[0] != w3.shape[0]:
            raise ValueError(f'Invalid shapes for w3: {w3.shape} / b3: {b3.shape}')
    if op is None:
        op = SwiGLUOpDispatch.from_arguments(x, w1, b1, w2, b2, w3, b3).op
    if not op.PACKED_WEIGHTS:
        return op(x, w1, b1, w2, b2, w3, b3).reshape([*batch_shape, -1])
    w1w2 = stack_or_none((w1, w2), dim=0)
    if b1 is not None and b2 is not None:
        b1b2: Optional[torch.Tensor] = stack_or_none((b1, b2), dim=0)
        if b1b2 is None:
            raise NotImplementedError('b1/b2 needs to be properly packed')
    else:
        b1b2 = None
        assert b1 is None and b2 is None
    if w1w2 is None:
        raise NotImplementedError('w1/w2 needs to be properly packed')
    return op(x, w1w2, b1b2, w3, b3).reshape([*batch_shape, -1])