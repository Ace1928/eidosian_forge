import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ReverseSequence(self) -> None:
    self._test_op_upgrade('ReverseSequence', 10, [[3, 4, 5], [4]], [[3, 4, 5]], [TensorProto.FLOAT, TensorProto.INT64])