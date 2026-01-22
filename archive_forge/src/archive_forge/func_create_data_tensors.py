from __future__ import annotations
import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import ModelProto, TensorProto, checker, helper, shape_inference
from onnx.external_data_helper import (
from onnx.numpy_helper import from_array, to_array
def create_data_tensors(self, tensors_data: list[tuple[list[Any], Any]]) -> list[TensorProto]:
    tensors = []
    for value, tensor_name in tensors_data:
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensors.append(tensor)
    return tensors