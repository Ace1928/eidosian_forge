import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GreaterOrEqual(self) -> None:
    self._test_op_upgrade('GreaterOrEqual', 12, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])