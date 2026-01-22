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
def test_protos_exist(self) -> None:
    _ = onnx.AttributeProto
    _ = onnx.NodeProto
    _ = onnx.GraphProto
    _ = onnx.ModelProto