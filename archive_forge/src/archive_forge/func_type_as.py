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
@_onnx_symbolic('aten::type_as')
@_beartype.beartype
def type_as(g: jit_utils.GraphContext, self, other):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    other_dtype = symbolic_helper._try_get_scalar_type(other)
    if self_dtype == other_dtype and self_dtype is not None:
        return self
    if other_dtype is not None:
        return g.op('Cast', self, to_i=other_dtype.onnx_type())
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at('type_as', self, other)
    raise errors.SymbolicValueError('Unsupported: ONNX export of type_as for tensor of unknown dtype. Please check if the dtype of the parameter passed to the type_as function is correct.', other)