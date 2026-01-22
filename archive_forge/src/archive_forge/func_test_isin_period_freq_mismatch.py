import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
def test_isin_period_freq_mismatch(self):
    dti = date_range('2013-01-01', '2013-01-05')
    pi = dti.to_period('M')
    ser = Series(pi)
    dtype = dti.to_period('Y').dtype
    other = PeriodArray._simple_new(pi.asi8, dtype=dtype)
    res = pi.isin(other)
    expected = np.array([False] * len(pi), dtype=bool)
    tm.assert_numpy_array_equal(res, expected)
    res = ser.isin(other)
    tm.assert_series_equal(res, Series(expected))
    res = pd.core.algorithms.isin(ser, other)
    tm.assert_numpy_array_equal(res, expected)