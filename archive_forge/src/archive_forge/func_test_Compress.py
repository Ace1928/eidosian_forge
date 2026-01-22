import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Compress(self) -> None:
    self._test_op_upgrade('Compress', 9, [[6, 7], [3]], [[3]], [TensorProto.FLOAT, TensorProto.BOOL], [TensorProto.FLOAT])