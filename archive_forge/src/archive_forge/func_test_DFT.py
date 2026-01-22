import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_DFT(self) -> None:
    self._test_op_upgrade('DFT', 17, [[2, 16, 1], []], [[2, 16, 2]])
    self._test_op_upgrade('DFT', 17, [[2, 16, 2], []], [[2, 16, 2]])
    self._test_op_upgrade('DFT', 17, [[2, 16, 1], []], [[2, 9, 2]], attrs={'onesided': 1})
    self._test_op_upgrade('DFT', 17, [[2, 16, 2], []], [[2, 9, 2]], attrs={'onesided': 1})
    self._test_op_upgrade('DFT', 17, [[2, 16, 1], []], [[2, 16, 2]], attrs={'inverse': 1})
    self._test_op_upgrade('DFT', 17, [[2, 16, 2], []], [[2, 16, 2]], attrs={'inverse': 1})
    self._test_op_upgrade('DFT', 17, [[2, 16, 2], []], [[2, 16, 2]], attrs={'inverse': 1, 'axis': 0})