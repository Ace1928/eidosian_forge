from __future__ import annotations
import dataclasses
import types
from typing import Optional, TYPE_CHECKING, Union
import torch._ops
from torch.onnx._internal import _beartype
@classmethod
@_beartype.beartype
def from_name_parts(cls, namespace: str, op_name: str, overload: Optional[str]=None) -> OpName:
    if overload is None or overload == '':
        overload = 'default'
    return cls(namespace, op_name, overload)