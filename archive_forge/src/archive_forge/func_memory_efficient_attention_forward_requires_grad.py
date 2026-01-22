from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def memory_efficient_attention_forward_requires_grad(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: Optional[Union[torch.Tensor, AttentionBias]]=None, p: float=0.0, scale: Optional[float]=None, *, op: Optional[Type[AttentionFwOpBase]]=None, output_dtype: Optional[torch.dtype]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple (output, lse), where `lse` can be used to compute the backward pass later.
    See :attr:`xformers.ops.memory_efficient_attention` for an explanation of the arguments
    See :attr:`xformers.ops.memory_efficient_attention_backward` for running the backward pass
    """
    if p != 0.0:
        raise NotImplementedError('dropout is not supported on the non-autograd API. If you want to use dropout, please call `memory_efficient_attention` directly')
    out, ctx = _memory_efficient_attention_forward_requires_grad(Inputs(query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale, output_dtype=output_dtype), op=op)
    return (out, ctx.lse)