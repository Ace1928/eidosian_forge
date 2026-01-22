import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Resize(self) -> None:
    self._test_op_upgrade('Resize', 10, [[3, 4, 5], [3]], [[3, 8, 15]])