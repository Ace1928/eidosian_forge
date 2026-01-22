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
def create_external_data_tensors(self, tensors_data: list[tuple[list[Any], Any]]) -> list[TensorProto]:
    tensor_filename = 'tensors.bin'
    tensors = []
    with open(os.path.join(self.temp_dir, tensor_filename), 'ab') as data_file:
        for value, tensor_name in tensors_data:
            tensor = from_array(np.array(value))
            offset = data_file.tell()
            if offset % 4096 != 0:
                data_file.write(b'\x00' * (4096 - offset % 4096))
                offset = offset + 4096 - offset % 4096
            data_file.write(tensor.raw_data)
            set_external_data(tensor, location=tensor_filename, offset=offset, length=data_file.tell() - offset)
            tensor.name = tensor_name
            tensor.ClearField('raw_data')
            tensor.data_location = onnx.TensorProto.EXTERNAL
            tensors.append(tensor)
    return tensors