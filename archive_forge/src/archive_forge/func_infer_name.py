from __future__ import annotations
import abc
from typing import Any, ClassVar, Iterable
import numpy as np
from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history, get_schema, onnx_opset_version
from onnx.helper import make_node, make_tensor_type_proto, np_dtype_to_tensor_dtype
from onnx.numpy_helper import to_array, unpack_int4
from onnx.onnx_pb import AttributeProto, GraphProto, NodeProto, TypeProto
from onnx.reference.custom_element_types import (
@classmethod
def infer_name(cls):
    name = cls.__name__
    if '_' not in name:
        return (name, onnx_opset_version())
    name, vers = name.rsplit('_', 1)
    try:
        i_vers = int(vers)
    except ValueError:
        return (cls.__name__, onnx_opset_version())
    return (name, i_vers)