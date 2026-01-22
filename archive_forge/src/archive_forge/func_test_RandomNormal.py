import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_RandomNormal(self) -> None:
    self._test_op_upgrade('RandomNormal', 1, [], [[3, 4, 5]], attrs={'shape': [3, 4, 5]})