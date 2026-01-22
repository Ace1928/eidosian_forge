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
@aten.tensor_split.tensor_indices_or_sections.py_impl(DispatchKey.CompositeImplicitAutograd)
def tensor_split_tensor_indices_or_sections_py_impl(self: Tensor, tensor_indices_or_sections: Tensor, dim: int=0) -> List[Tensor]:
    assert tensor_indices_or_sections.device.type == 'cpu'
    assert tensor_indices_or_sections.dtype == torch.int64
    split_dim = tensor_indices_or_sections.dim()
    torch._check(split_dim == 1 or split_dim == 0, lambda: f'tensor_split expected tensor_indices_or_sections to be a zero-dimensional or one-dimensional tensor, but got a tensor with {split_dim} dims')
    if split_dim == 0:
        sections = tensor_indices_or_sections.item()
        assert isinstance(sections, IntLike)
        return self.tensor_split(sections, dim)
    else:
        indices = [i.item() for i in tensor_indices_or_sections]
        return self.tensor_split(indices, dim)