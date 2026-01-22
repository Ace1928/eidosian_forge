import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition(aten.multilabel_margin_loss_forward)
@aten.multilabel_margin_loss_forward.default.py_impl(DispatchKey.Autograd)
@out_wrapper('output', 'is_target')
def multilabel_margin_loss_forward(input: Tensor, target: Tensor, reduction: int) -> Tuple[Tensor, Tensor]:
    orig_input_shape = input.shape
    orig_target_shape = target.shape
    input = torch.atleast_2d(input)
    target = torch.atleast_2d(target)
    dim = input.shape[1]
    torch._check(len(orig_input_shape) <= 2 and dim != 0, lambda: f'Expected non-empty vector or matrix with optional 0-dim batch size, but got: {orig_input_shape}')
    torch._check(len(orig_target_shape) <= 2 and orig_target_shape == orig_input_shape, lambda: f'inconsistent target size: {orig_target_shape} for input of size: {orig_input_shape}')
    idx = torch.arange(dim, device=target.device)
    is_end = target == -1
    end_idx = torch.amin(torch.where(is_end, idx, dim), dim=-1, keepdim=True)
    target_mask = idx < end_idx
    tidx0 = torch.where(target_mask, target, 0)
    u = torch.gather(input, dim=-1, index=tidx0)
    tidx1 = torch.where(target_mask, target, -1)
    is_target = torch.any(idx == tidx1.unsqueeze(dim=-1), dim=1)
    z = 1.0 - u.T.unsqueeze(dim=-1) + input
    z = z.clamp_min(0)
    z = z / dim
    z = torch.where(is_target, 0, z)
    if reduction == Reduction.MEAN.value:
        z = z.sum(dim=(0, -1)).mean()
    elif reduction == Reduction.SUM.value:
        z = z.sum()
    else:
        z = z.sum(dim=(0, -1))
    is_target = is_target.to(input.dtype).reshape(orig_target_shape)
    return (z, is_target)