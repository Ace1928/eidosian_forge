import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_factorize_dst(self, index_or_series):
    idx = date_range('2016-11-06', freq='h', periods=12, tz='US/Eastern')
    obj = index_or_series(idx)
    arr, res = obj.factorize()
    tm.assert_numpy_array_equal(arr, np.arange(12, dtype=np.intp))
    tm.assert_index_equal(res, idx)
    if index_or_series is Index:
        assert res.freq == idx.freq
    idx = date_range('2016-06-13', freq='h', periods=12, tz='US/Eastern')
    obj = index_or_series(idx)
    arr, res = obj.factorize()
    tm.assert_numpy_array_equal(arr, np.arange(12, dtype=np.intp))
    tm.assert_index_equal(res, idx)
    if index_or_series is Index:
        assert res.freq == idx.freq