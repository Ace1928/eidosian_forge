import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_StringConcat(self) -> None:
    self._test_op_upgrade('StringConcat', 20, [[2, 3], [2, 3]], [[2, 3]])