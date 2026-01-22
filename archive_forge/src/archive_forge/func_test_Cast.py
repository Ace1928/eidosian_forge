import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Cast(self) -> None:
    self._test_op_upgrade('Cast', 6, [[2, 3]], [[2, 3]], [TensorProto.INT64], attrs={'to': 1})