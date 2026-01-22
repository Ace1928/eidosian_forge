import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Det_1(self) -> None:
    self._test_op_upgrade('Det', 11, [[3, 5, 5]], [[3]])