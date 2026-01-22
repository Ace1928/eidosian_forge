import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Split(self) -> None:
    self._test_op_upgrade('Split', 2, [[3, 4, 7]], [[3, 4, 2], [3, 4, 1], [3, 4, 4]], attrs={'axis': 2, 'split': [2, 1, 4]})