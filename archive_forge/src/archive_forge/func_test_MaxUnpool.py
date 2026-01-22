import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_MaxUnpool(self) -> None:
    self._test_op_upgrade('MaxUnpool', 9, [[1, 1, 5, 5], [1, 1, 5, 5]], [[1, 1, 6, 6]], [TensorProto.FLOAT, TensorProto.INT64], attrs={'kernel_shape': [2, 2]})