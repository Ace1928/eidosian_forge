import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Col2Im_5D(self) -> None:
    self._test_op_upgrade('Col2Im', 18, [[1, 10, 12], [3], [3]], [[1, 2, 3, 4, 5]])