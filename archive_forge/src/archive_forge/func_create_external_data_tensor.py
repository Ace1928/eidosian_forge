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
def create_external_data_tensor(self, value: list[Any], tensor_name: str, location: str='') -> TensorProto:
    tensor = from_array(np.array(value))
    tensor.name = tensor_name
    tensor_filename = location or f'{tensor_name}.bin'
    set_external_data(tensor, location=tensor_filename)
    tensor.ClearField('raw_data')
    tensor.data_location = onnx.TensorProto.EXTERNAL
    return tensor