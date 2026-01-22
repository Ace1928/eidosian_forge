import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8_e4m3fn_inf(self):
    x = np.float32(np.inf)
    to = helper.float32_to_float8e4m3(x)
    back = numpy_helper.float8e4m3_to_float32(to)
    self.assertEqual(back, 448)
    x = np.float32(np.inf)
    to = helper.float32_to_float8e4m3(x, saturate=False)
    back = numpy_helper.float8e4m3_to_float32(to)
    self.assertTrue(np.isnan(back))
    x = np.float32(-np.inf)
    to = helper.float32_to_float8e4m3(x)
    self.assertEqual(to & 128, 128)
    back = numpy_helper.float8e4m3_to_float32(to)
    self.assertEqual(back, -448)
    x = np.float32(-np.inf)
    to = helper.float32_to_float8e4m3(x, saturate=False)
    self.assertEqual(to & 128, 128)
    back = numpy_helper.float8e4m3_to_float32(to)
    self.assertTrue(np.isnan(back))