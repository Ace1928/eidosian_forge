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
def test_save_and_load_model_when_input_is_file_name(self) -> None:
    proto = _simple_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model.onnx')
        onnx.save_model(proto, model_path, format=self.format)
        loaded_proto = onnx.load_model(model_path, format=self.format)
        self.assertEqual(proto, loaded_proto)