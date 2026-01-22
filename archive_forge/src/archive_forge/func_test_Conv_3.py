import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Conv_3(self) -> None:
    self._test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 1, 2, 2], [4]], [[1, 4, 3, 7]], attrs={'dilations': [1, 2], 'group': 3, 'pads': [0, 1, 2, 3], 'strides': [2, 1]})