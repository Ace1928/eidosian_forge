import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Concat(self) -> None:
    self._test_op_upgrade('Concat', 1, [[2, 3], [2, 4]], [[2, 7]])