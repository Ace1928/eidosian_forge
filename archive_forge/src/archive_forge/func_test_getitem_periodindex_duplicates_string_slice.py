import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_getitem_periodindex_duplicates_string_slice(self, using_copy_on_write, warn_copy_on_write):
    idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq='Y-JUN')
    ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)
    original = ts.copy()
    result = ts['2007']
    expected = ts[1:3]
    tm.assert_series_equal(result, expected)
    with tm.assert_cow_warning(warn_copy_on_write):
        result[:] = 1
    if using_copy_on_write:
        tm.assert_series_equal(ts, original)
    else:
        assert (ts[1:3] == 1).all()
    idx = PeriodIndex([2000, 2007, 2007, 2009, 2007], freq='Y-JUN')
    ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)
    result = ts['2007']
    expected = ts[idx == '2007']
    tm.assert_series_equal(result, expected)