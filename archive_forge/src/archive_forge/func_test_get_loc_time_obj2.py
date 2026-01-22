from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('offset', [-10, 10])
def test_get_loc_time_obj2(self, monkeypatch, offset):
    size_cutoff = 50
    n = size_cutoff + offset
    key = time(15, 11, 30)
    start = key.hour * 3600 + key.minute * 60 + key.second
    step = 24 * 3600
    with monkeypatch.context():
        monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
        idx = date_range('2014-11-26', periods=n, freq='s')
        ts = pd.Series(np.random.default_rng(2).standard_normal(n), index=idx)
        locs = np.arange(start, n, step, dtype=np.intp)
        result = ts.index.get_loc(key)
        tm.assert_numpy_array_equal(result, locs)
        tm.assert_series_equal(ts[key], ts.iloc[locs])
        left, right = (ts.copy(), ts.copy())
        left[key] *= -10
        right.iloc[locs] *= -10
        tm.assert_series_equal(left, right)