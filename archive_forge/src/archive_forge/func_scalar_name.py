from __future__ import annotations
import enum
import typing
from typing import Dict, Literal, Optional, Union
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype
@_beartype.beartype
def scalar_name(self) -> ScalarName:
    """Convert a JitScalarType to a JIT scalar type name."""
    return _SCALAR_TYPE_TO_NAME[self]