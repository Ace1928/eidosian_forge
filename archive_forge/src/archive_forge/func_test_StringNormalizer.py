import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_StringNormalizer(self) -> None:
    self._test_op_upgrade('StringNormalizer', 10, [[1, 3]], [[1, 3]], [TensorProto.STRING], [TensorProto.STRING], attrs={'case_change_action': 'LOWER'})