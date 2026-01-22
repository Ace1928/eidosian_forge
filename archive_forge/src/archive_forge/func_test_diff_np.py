import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_diff_np(self):
    ser = Series(np.arange(5))
    res = np.diff(ser)
    expected = np.array([1, 1, 1, 1])
    tm.assert_numpy_array_equal(res, expected)