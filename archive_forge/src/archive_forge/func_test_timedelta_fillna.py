from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_timedelta_fillna(self, frame_or_series, unit):
    ser = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')], dtype=f'M8[{unit}]')
    td = ser.diff()
    obj = frame_or_series(td).copy()
    result = obj.fillna(Timedelta(seconds=0))
    expected = Series([timedelta(0), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    res = obj.fillna(1)
    expected = obj.astype(object).fillna(1)
    tm.assert_equal(res, expected)
    result = obj.fillna(Timedelta(seconds=1))
    expected = Series([timedelta(seconds=1), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    result = obj.fillna(timedelta(days=1, seconds=1))
    expected = Series([timedelta(days=1, seconds=1), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    result = obj.fillna(np.timedelta64(10 ** 9))
    expected = Series([timedelta(seconds=1), timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    result = obj.fillna(NaT)
    expected = Series([NaT, timedelta(0), timedelta(1), timedelta(days=1, seconds=9 * 3600 + 60 + 1)], dtype=f'm8[{unit}]')
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    td[2] = np.nan
    obj = frame_or_series(td).copy()
    result = obj.ffill()
    expected = td.fillna(Timedelta(seconds=0))
    expected[0] = np.nan
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    td[2] = np.nan
    obj = frame_or_series(td)
    result = obj.bfill()
    expected = td.fillna(Timedelta(seconds=0))
    expected[2] = timedelta(days=1, seconds=9 * 3600 + 60 + 1)
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)