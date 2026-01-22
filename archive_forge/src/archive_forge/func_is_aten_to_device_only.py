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
@_beartype.beartype
def is_aten_to_device_only(args):
    if len(args) == 4:
        return args[0].node().kind() == 'prim::device' or args[0].type().isSubtypeOf(_C.ListType.ofInts()) or isinstance(args[0].type(), _C.DeviceObjType)
    elif len(args) == 5:
        dtype = symbolic_helper._get_const(args[1], 'i', 'dtype')
        return dtype is None
    elif len(args) in (6, 7):
        dtype = symbolic_helper._get_const(args[0], 'i', 'dtype')
        return dtype is None
    return False