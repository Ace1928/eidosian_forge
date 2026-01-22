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
@_onnx_symbolic('prim::tolist')
@_beartype.beartype
def prim_tolist(g: jit_utils.GraphContext, input, dim_val, elem_ty_val):
    """tolist is currently supported only for 1D input tensors.

    dim_val and elem_ty_val represent dimension and type annotations
    that need to match dimension and type of the input tensor.
    """
    dim = symbolic_helper._maybe_get_const(dim_val, 'i')
    if dim > 1:
        return symbolic_helper._unimplemented('prim::tolist', 'dim_val > 1', input)
    return input