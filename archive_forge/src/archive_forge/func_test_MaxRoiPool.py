import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_MaxRoiPool(self) -> None:
    self._test_op_upgrade('MaxRoiPool', 1, [[2, 3, 20, 20], [4, 5]], [[4, 3, 3, 3]], attrs={'pooled_shape': [3, 3]})