import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Reshape(self) -> None:
    self._test_op_upgrade('Reshape', 1, [[3, 4, 5]], [[3, 10, 2]], attrs={'consumed_inputs': [0], 'shape': [3, 10, 2]})