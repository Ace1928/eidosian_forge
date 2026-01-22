from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_dt64tz_with_method(self):
    ser = Series([Timestamp('2012-11-11 00:00:00+01:00'), NaT])
    exp = Series([Timestamp('2012-11-11 00:00:00+01:00'), Timestamp('2012-11-11 00:00:00+01:00')])
    tm.assert_series_equal(ser.fillna(method='pad'), exp)
    ser = Series([NaT, Timestamp('2012-11-11 00:00:00+01:00')])
    exp = Series([Timestamp('2012-11-11 00:00:00+01:00'), Timestamp('2012-11-11 00:00:00+01:00')])
    tm.assert_series_equal(ser.fillna(method='bfill'), exp)