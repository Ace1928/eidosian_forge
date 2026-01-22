from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def make_large_tensor_proto(location: str, tensor_name: str, tensor_type: int, shape: tuple[int, ...]) -> onnx.TensorProto:
    """Create an external tensor.

    Arguments:
        location: unique identifier (not necessary a path)
        tensor_name: tensor name in the graph
        tensor_type: onnx type
        shape: shape the of the initializer

    Returns:
        the created tensor
    """
    tensor_location = location
    tensor = onnx.TensorProto()
    tensor.name = tensor_name
    _set_external_data(tensor, tensor_location)
    tensor.data_type = tensor_type
    tensor.dims.extend(shape)
    return tensor