import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_from_dict_differing_value_types(self):
    with self.assertRaises(TypeError):
        numpy_helper.from_dict({0: np.array(1), 1: np.array(0.9)})