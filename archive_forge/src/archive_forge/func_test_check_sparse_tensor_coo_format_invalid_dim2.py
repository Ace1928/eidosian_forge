import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_sparse_tensor_coo_format_invalid_dim2(self) -> None:
    sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 1], [0, 1, 2])
    self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)