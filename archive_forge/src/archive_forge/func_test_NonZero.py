import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_NonZero(self) -> None:
    self._test_op_upgrade('NonZero', 9, [[3, 3]], [[2, 4]], output_types=[TensorProto.INT64])