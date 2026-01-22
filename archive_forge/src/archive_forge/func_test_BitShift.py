import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_BitShift(self) -> None:
    self._test_op_upgrade('BitShift', 11, [[2, 3], [2, 3]], [[2, 3]], [TensorProto.UINT8, TensorProto.UINT8], [TensorProto.UINT8], attrs={'direction': 'RIGHT'})