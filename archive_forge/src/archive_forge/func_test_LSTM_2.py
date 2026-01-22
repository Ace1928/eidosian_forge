import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_LSTM_2(self) -> None:
    self._test_op_upgrade('LSTM', 7, [[5, 3, 4], [2, 24, 4], [2, 24, 4]], [[5, 2, 3, 6], [2, 3, 6], [2, 3, 6]], attrs={'hidden_size': 6, 'direction': 'bidirectional'})