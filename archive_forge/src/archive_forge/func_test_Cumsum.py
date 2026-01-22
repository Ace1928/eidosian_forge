import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Cumsum(self) -> None:
    self._test_op_upgrade('CumSum', 11, [[3, 4, 5], []], [[3, 4, 5]], [TensorProto.FLOAT, TensorProto.INT64])