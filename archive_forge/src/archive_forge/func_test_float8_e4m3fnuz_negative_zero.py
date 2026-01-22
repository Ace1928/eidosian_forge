import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8_e4m3fnuz_negative_zero(self):
    x = numpy_helper.float8e5m2_to_float32(128)
    to = helper.float32_to_float8e4m3(x, uz=True)
    self.assertEqual(to, 0)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertEqual(back, 0)
    x = numpy_helper.float8e5m2_to_float32(128)
    to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertEqual(back, 0)
    self.assertEqual(to, 0)