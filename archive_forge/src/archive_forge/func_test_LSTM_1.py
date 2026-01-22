import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_LSTM_1(self) -> None:
    self._test_op_upgrade('LSTM', 7, [[5, 3, 4], [1, 24, 4], [1, 24, 4]], [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]], attrs={'hidden_size': 6})