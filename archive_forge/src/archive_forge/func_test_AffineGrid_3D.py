import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_AffineGrid_3D(self) -> None:
    N, _, D, H, W = (2, 3, 4, 5, 6)
    self._test_op_upgrade('AffineGrid', 20, [[N, 3, 4], [5]], [[N, D, H, W, 3]])