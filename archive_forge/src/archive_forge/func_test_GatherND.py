import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GatherND(self) -> None:
    self._test_op_upgrade('GatherND', 11, [[1, 2, 3], [1, 2, 3]], [[1, 2]])