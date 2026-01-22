import unittest
import automatic_conversion_test_base
import numpy as np
import parameterized
import onnx
from onnx import helper
def test_dft20_initializer_axis(self) -> None:
    self._test_model_conversion(to_opset=19, model='\n            <ir_version: 9, opset_import: [ "" : 20]>\n            dft_no_axis (float[N, M, 1] x, int64 dft_length) => (float[N, K, 2] y)\n            <int64 axis = {1}>\n            {\n                y = DFT (x, dft_length, axis)\n            }\n        ')