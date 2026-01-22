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
@_onnx_symbolic('prim::ListUnpack')
@_beartype.beartype
def prim_list_unpack(g: jit_utils.GraphContext, *inputs, **kwargs) -> Optional[List[_C.Value]]:
    if len(inputs) == 1 and inputs[0].node().kind() == 'prim::ListConstruct':
        return symbolic_helper._unpack_list(inputs[0])
    return None