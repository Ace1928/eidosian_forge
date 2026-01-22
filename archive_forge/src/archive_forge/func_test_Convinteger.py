import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Convinteger(self) -> None:
    self._test_op_upgrade('ConvInteger', 10, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]], [TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8], [TensorProto.INT32])