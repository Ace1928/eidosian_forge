from __future__ import annotations
import enum
import typing
from typing import Dict, Literal, Optional, Union
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype
@_beartype.beartype
def valid_torch_name(torch_name: Union[TorchName, str]) -> bool:
    """Return whether the given torch name is a valid torch type name."""
    return torch_name in _TORCH_NAME_TO_SCALAR_TYPE