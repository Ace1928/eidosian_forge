import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.unbind.int, 'self: jt_all, dim: any?')
def unbind_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    dim = new_kwargs['dim']
    if dim != 0:
        raise RuntimeError('unbind(): only supported for NestedTensor on dim=0')
    inp = new_kwargs.pop('input')
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()
    if inp._ragged_idx != 1:
        raise RuntimeError('unbind(): only supported for NestedTensor when jagged dimension is 1')
    if lengths is None:
        return torch.split(values, offsets.diff().tolist())
    return [values[offsets[i]:offsets[i] + lengths[i]] for i in range(lengths.shape[0])]