from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::linalg_norm')
@symbolic_helper.parse_args('v', 'v', 'is', 'b', 'v')
@_beartype.beartype
def linalg_norm(g: jit_utils.GraphContext, self: torch._C.Value, ord: torch._C.Value, dim: Optional[Sequence[int]], keepdim: bool, dtype: torch._C.Value):
    ord_value = None
    if dim is None:
        if symbolic_helper._is_none(ord):
            self = symbolic_helper._reshape_helper(g, self, [-1])
            ord = g.op('Constant', value_t=torch.LongTensor([2]))
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented('dim', 'Input rank must be known at export time.', self)
        if self_dim == 1:
            ord_value = symbolic_helper._parse_arg(ord, 'f')
        else:
            dim = [0, 1]
    elif len(dim) == 1:
        if symbolic_helper._is_none(ord):
            ord = g.op('Constant', value_t=torch.LongTensor([2]))
        ord_value = symbolic_helper._parse_arg(ord, 'f')
    if ord_value:
        return linalg_vector_norm(g, self, ord_value, dim, keepdim, dtype)
    return linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)