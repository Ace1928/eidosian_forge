from __future__ import annotations
import enum
import typing
from typing import Dict, Literal, Optional, Union
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype
@_beartype.beartype
def torch_name(self) -> TorchName:
    """Convert a JitScalarType to a torch type name."""
    return _SCALAR_TYPE_TO_TORCH_NAME[self]