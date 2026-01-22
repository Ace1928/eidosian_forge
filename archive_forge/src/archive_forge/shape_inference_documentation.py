from __future__ import annotations
import os
from typing import Sequence
import onnx
import onnx.onnx_cpp2py_export.shape_inference as C  # noqa: N812
from onnx import AttributeProto, FunctionProto, ModelProto, TypeProto
Apply type-and-shape-inference to given function body, with given input types
    and given input attribute values.
    