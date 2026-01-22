import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_InstanceNormalization(self) -> None:
    self._test_op_upgrade('InstanceNormalization', 1, [[1, 3], [3], [3]], [[1, 3]], attrs={'consumed_inputs': [0]})