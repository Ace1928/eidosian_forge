import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_attr_default_value(self) -> None:
    v = defs.get_schema('BatchNormalization').attributes['epsilon'].default_value
    self.assertEqual(type(v), onnx.AttributeProto)
    self.assertEqual(v.type, onnx.AttributeProto.FLOAT)