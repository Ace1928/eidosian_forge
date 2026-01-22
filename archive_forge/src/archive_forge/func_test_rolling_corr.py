import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_rolling_corr(series):
    A = series
    B = A + np.random.default_rng(2).standard_normal(len(A))
    result = A.rolling(window=50, min_periods=25).corr(B)
    tm.assert_almost_equal(result.iloc[-1], np.corrcoef(A[-50:], B[-50:])[0, 1])