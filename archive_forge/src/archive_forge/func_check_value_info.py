from __future__ import annotations
import os
import sys
from typing import Any, Callable, TypeVar
from google.protobuf.message import Message
import onnx.defs
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx.shape_inference
from onnx import (
def check_value_info(value_info: ValueInfoProto, ctx: C.CheckerContext=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(value_info, ValueInfoProto)
    return C.check_value_info(value_info.SerializeToString(), ctx)