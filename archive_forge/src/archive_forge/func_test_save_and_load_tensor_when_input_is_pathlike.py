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
def test_save_and_load_tensor_when_input_is_pathlike(self) -> None:
    proto = _simple_tensor()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = pathlib.Path(temp_dir, 'model.onnx')
        onnx.save_tensor(proto, model_path, format=self.format)
        loaded_proto = onnx.load_tensor(model_path, format=self.format)
        self.assertEqual(proto, loaded_proto)