import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_diff_tz(self):
    ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    ts.diff()
    result = ts.diff(-1)
    expected = ts - ts.shift(-1)
    tm.assert_series_equal(result, expected)
    result = ts.diff(0)
    expected = ts - ts
    tm.assert_series_equal(result, expected)