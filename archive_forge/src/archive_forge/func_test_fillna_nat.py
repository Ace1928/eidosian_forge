from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_nat(self):
    series = Series([0, 1, 2, NaT._value], dtype='M8[ns]')
    filled = series.fillna(method='pad')
    filled2 = series.fillna(value=series.values[2])
    expected = series.copy()
    expected.iloc[3] = expected.iloc[2]
    tm.assert_series_equal(filled, expected)
    tm.assert_series_equal(filled2, expected)
    df = DataFrame({'A': series})
    filled = df.fillna(method='pad')
    filled2 = df.fillna(value=series.values[2])
    expected = DataFrame({'A': expected})
    tm.assert_frame_equal(filled, expected)
    tm.assert_frame_equal(filled2, expected)
    series = Series([NaT._value, 0, 1, 2], dtype='M8[ns]')
    filled = series.fillna(method='bfill')
    filled2 = series.fillna(value=series[1])
    expected = series.copy()
    expected[0] = expected[1]
    tm.assert_series_equal(filled, expected)
    tm.assert_series_equal(filled2, expected)
    df = DataFrame({'A': series})
    filled = df.fillna(method='bfill')
    filled2 = df.fillna(value=series[1])
    expected = DataFrame({'A': expected})
    tm.assert_frame_equal(filled, expected)
    tm.assert_frame_equal(filled2, expected)