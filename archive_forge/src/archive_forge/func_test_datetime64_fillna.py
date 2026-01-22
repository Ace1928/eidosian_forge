from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_datetime64_fillna(self):
    ser = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')])
    ser[2] = np.nan
    result = ser.ffill()
    expected = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130103 9:01:01')])
    tm.assert_series_equal(result, expected)
    result = ser.bfill()
    expected = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130103 9:01:01'), Timestamp('20130103 9:01:01')])
    tm.assert_series_equal(result, expected)