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
@_onnx_symbolic('prim::dtype')
@_beartype.beartype
def prim_dtype(g: jit_utils.GraphContext, self):
    scalar_type = symbolic_helper._try_get_scalar_type(self)
    if scalar_type is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    return g.op('Constant', value_t=torch.tensor(scalar_type))