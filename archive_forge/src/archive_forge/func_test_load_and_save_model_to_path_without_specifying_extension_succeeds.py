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
def test_load_and_save_model_to_path_without_specifying_extension_succeeds(self) -> None:
    proto = _simple_model()
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model')
        onnx.save_model(proto, model_path, format='textproto')
        with self.assertRaises(google.protobuf.message.DecodeError):
            onnx.load_model(model_path)
        loaded_proto = onnx.load_model(model_path, format='textproto')
        self.assertEqual(proto, loaded_proto)