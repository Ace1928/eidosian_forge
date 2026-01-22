import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ScatterElements_1(self) -> None:
    self._test_op_upgrade('ScatterElements', 11, [[2, 3], [1, 2], [1, 2]], [[2, 3]], [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], [TensorProto.FLOAT])