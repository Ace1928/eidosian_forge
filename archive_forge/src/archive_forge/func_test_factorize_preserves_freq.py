import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_factorize_preserves_freq(self):
    idx3 = date_range('2000-01', periods=4, freq='ME', tz='Asia/Tokyo')
    exp_arr = np.array([0, 1, 2, 3], dtype=np.intp)
    arr, idx = idx3.factorize()
    tm.assert_numpy_array_equal(arr, exp_arr)
    tm.assert_index_equal(idx, idx3)
    assert idx.freq == idx3.freq
    arr, idx = factorize(idx3)
    tm.assert_numpy_array_equal(arr, exp_arr)
    tm.assert_index_equal(idx, idx3)
    assert idx.freq == idx3.freq