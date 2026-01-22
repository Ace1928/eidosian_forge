from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def memory_efficient_attention_backward(grad: torch.Tensor, output: torch.Tensor, lse: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_bias: Optional[Union[torch.Tensor, AttentionBias]]=None, p: float=0.0, scale: Optional[float]=None, *, op: Optional[Type[AttentionBwOpBase]]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the attention.
    Returns a tuple (dq, dk, dv)
    See :attr:`xformers.ops.memory_efficient_attention` for an explanation of the arguments.
    `lse` is the tensor returned by :attr:`xformers.ops.memory_efficient_attention_forward_requires_grad`
    """
    if p != 0.0:
        raise NotImplementedError('dropout is not supported on the non-autograd API. If you want to use dropout, please call `memory_efficient_attention` directly')
    gradients = _memory_efficient_attention_backward(Context(out=output, lse=lse), Inputs(query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale), grad, op=op)
    return (gradients.dq, gradients.dk, gradients.dv)