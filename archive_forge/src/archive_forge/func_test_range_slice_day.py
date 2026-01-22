import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('make_range', [date_range, period_range])
def test_range_slice_day(self, make_range):
    idx = make_range(start='2013/01/01', freq='D', periods=400)
    msg = 'slice indices must be integers or None or have an __index__ method'
    values = ['2014', '2013/02', '2013/01/02', '2013/02/01 9H', '2013/02/01 09:00']
    for v in values:
        with pytest.raises(TypeError, match=msg):
            idx[v:]
    s = Series(np.random.default_rng(2).random(len(idx)), index=idx)
    tm.assert_series_equal(s['2013/01/02':], s[1:])
    tm.assert_series_equal(s['2013/01/02':'2013/01/05'], s[1:5])
    tm.assert_series_equal(s['2013/02':], s[31:])
    tm.assert_series_equal(s['2014':], s[365:])
    invalid = ['2013/02/01 9H', '2013/02/01 09:00']
    for v in invalid:
        with pytest.raises(TypeError, match=msg):
            idx[v:]