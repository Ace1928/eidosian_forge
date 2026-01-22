import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Flatten(self) -> None:
    self._test_op_upgrade('Flatten', 1, [[3, 4, 5]], [[3, 20]], attrs={'axis': 1})