import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Acos(self) -> None:
    self._test_op_upgrade('Acos', 7)