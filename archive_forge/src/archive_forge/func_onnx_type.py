from __future__ import annotations
import enum
import typing
from typing import Dict, Literal, Optional, Union
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype
@_beartype.beartype
def onnx_type(self) -> _C_onnx.TensorProtoDataType:
    """Convert a JitScalarType to an ONNX data type."""
    if self not in _SCALAR_TYPE_TO_ONNX:
        raise errors.OnnxExporterError(f'Scalar type {self} cannot be converted to ONNX')
    return _SCALAR_TYPE_TO_ONNX[self]