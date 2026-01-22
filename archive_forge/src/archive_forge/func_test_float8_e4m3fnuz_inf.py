import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8_e4m3fnuz_inf(self):
    x = np.float32(np.inf)
    to = helper.float32_to_float8e4m3(x, uz=True)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertEqual(back, 240)
    x = np.float32(np.inf)
    to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertTrue(np.isnan(back))
    x = np.float32(-np.inf)
    to = helper.float32_to_float8e4m3(x, uz=True)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertEqual(back, -240)
    x = np.float32(-np.inf)
    to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertTrue(np.isnan(back))