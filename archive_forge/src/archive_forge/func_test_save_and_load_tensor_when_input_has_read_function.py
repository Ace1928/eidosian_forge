import io
import os
import pathlib
import tempfile
import unittest
import google.protobuf.message
import google.protobuf.text_format
import parameterized
import onnx
from onnx import serialization
def test_save_and_load_tensor_when_input_has_read_function(self) -> None:
    proto = _simple_tensor()
    f = io.BytesIO()
    onnx.save_tensor(proto, f, format=self.format)
    loaded_proto = onnx.load_tensor(io.BytesIO(f.getvalue()), format=self.format)
    self.assertEqual(proto, loaded_proto)