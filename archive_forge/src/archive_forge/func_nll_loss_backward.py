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
@register_decomposition(aten.nll_loss_backward)
@out_wrapper('grad_input')
def nll_loss_backward(grad_output: Tensor, self: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, ignore_index: int, total_weight: Tensor) -> Tensor:
    assert 0 <= self.dim() <= 2, 'input tensor should be 1D or 2D'
    assert target.dim() <= 1, '0D or 1D target tensor expected, multi-target not supported'
    no_batch_dim = self.dim() == 1 and target.dim() == 0
    assert no_batch_dim or self.shape[0] == target.shape[0], f'size mismatch (got input: {self.shape}, target: {target.shape})'
    assert total_weight.numel() == 1, ('expected total_weight to be a single element tensor, got: ', f'{total_weight.shape} ({total_weight.numel()} elements)')
    assert weight is None or weight.numel() == self.shape[-1], 'weight tensor should be defined either for all or no classes'
    if reduction == Reduction.NONE.value and self.dim() == 2:
        assert grad_output.dim() == 1 and grad_output.shape[0] == self.shape[0], f'Expected a tensor of dimension 1 and tensor.size[0] == {self.shape[0]} but got: dimension {grad_output.dim()} and tensor.size[0] == {grad_output.shape[0]}'
    else:
        assert grad_output.dim() <= 1 and grad_output.numel() == 1, f'Expected a single element grad_output tensor, but got: {grad_output.shape}'
    return _nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight)