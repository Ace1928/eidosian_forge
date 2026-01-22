import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_make_float8e5m2fnuz_tensor(self) -> None:
    y = helper.make_tensor('zero_point', TensorProto.FLOAT8E5M2FNUZ, [7], [0, 0.5, 1, 50000, 96, -1e-07, 1e-07])
    ynp = numpy_helper.to_array(y)
    expected = np.array([0, 0.5, 1, 49152, 96, 0, 0], dtype=np.float32)
    np.testing.assert_equal(expected, ynp)