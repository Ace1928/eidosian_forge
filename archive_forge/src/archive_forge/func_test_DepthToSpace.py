import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_DepthToSpace(self) -> None:
    self._test_op_upgrade('DepthToSpace', 1, [[1, 8, 3, 3]], [[1, 2, 6, 6]], attrs={'blocksize': 2})