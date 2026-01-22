import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_DynamicQuantizeLinear(self) -> None:
    self._test_op_upgrade('DynamicQuantizeLinear', 11, [[3, 4, 5]], [[3, 4, 5], [], []], output_types=[TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8])