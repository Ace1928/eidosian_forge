import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_rolling_min_min_periods(step):
    a = Series([1, 2, 3, 4, 5])
    result = a.rolling(window=100, min_periods=1, step=step).min()
    expected = Series(np.ones(len(a)))[::step]
    tm.assert_series_equal(result, expected)
    msg = 'min_periods 5 must be <= window 3'
    with pytest.raises(ValueError, match=msg):
        Series([1, 2, 3]).rolling(window=3, min_periods=5, step=step).min()