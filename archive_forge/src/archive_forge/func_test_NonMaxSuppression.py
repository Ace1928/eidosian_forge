import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_NonMaxSuppression(self) -> None:
    self._test_op_upgrade('NonMaxSuppression', 10, [[2, 3, 4], [3, 5, 6]], [[2, 3]], output_types=[TensorProto.INT64])