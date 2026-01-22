import operator
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.ops.array_ops import (
def test_object_comparison_2d():
    left = np.arange(9).reshape(3, 3).astype(object)
    right = left.T
    result = comparison_op(left, right, operator.eq)
    expected = np.eye(3).astype(bool)
    tm.assert_numpy_array_equal(result, expected)
    right.flags.writeable = False
    result = comparison_op(left, right, operator.ne)
    tm.assert_numpy_array_equal(result, ~expected)