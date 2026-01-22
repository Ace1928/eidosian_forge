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
@parameterized.parameterized.expand(itertools.product((TensorProto.UINT4, TensorProto.INT4), ((5, 4, 6), (4, 6, 5), (3, 3), (1,), (2 ** 10,))))
@unittest.skipIf(version_utils.numpy_older_than('1.22.0'), 'The test requires numpy 1.22.0 or later')
def test_make_4bit_tensor(self, dtype, dims) -> None:
    type_range = {TensorProto.UINT4: (0, 15), TensorProto.INT4: (-8, 7)}
    data = np.random.randint(type_range[dtype][0], high=type_range[dtype][1] + 1, size=dims)
    y = helper.make_tensor('y', dtype, data.shape, data)
    ynp = to_array_extended(y)
    np.testing.assert_equal(data, ynp)