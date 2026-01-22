import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GlobalAveragePool(self) -> None:
    self._test_op_upgrade('GlobalAveragePool', 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])