import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_QuantizeLinear(self) -> None:
    self._test_op_upgrade('QuantizeLinear', 10, [[3, 4, 5], [], []], [[3, 4, 5]], [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.UINT8], [TensorProto.UINT8])