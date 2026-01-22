from __future__ import annotations
import dataclasses
import types
from typing import Optional, TYPE_CHECKING, Union
import torch._ops
from torch.onnx._internal import _beartype
@classmethod
@_beartype.beartype
def from_op_overload(cls, op_overload: torch._ops.OpOverload) -> OpName:
    return cls.from_qualified_name(op_overload.name())