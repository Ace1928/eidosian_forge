import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Shape(self) -> None:
    self._test_op_upgrade('Shape', 1, [[3, 4, 5]], [[3]], output_types=[TensorProto.INT64])