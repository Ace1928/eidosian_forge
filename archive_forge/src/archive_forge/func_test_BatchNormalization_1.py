import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_BatchNormalization_1(self) -> None:
    self._test_op_upgrade('BatchNormalization', 1, [[1, 3], [3], [3], [3], [3]], [[1, 3]], attrs={'consumed_inputs': [1, 1], 'is_test': 1, 'spatial': 1})