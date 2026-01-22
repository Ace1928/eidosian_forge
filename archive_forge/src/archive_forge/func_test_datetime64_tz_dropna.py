import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_datetime64_tz_dropna(self, unit):
    ser = Series([Timestamp('2011-01-01 10:00'), NaT, Timestamp('2011-01-03 10:00'), NaT], dtype=f'M8[{unit}]')
    result = ser.dropna()
    expected = Series([Timestamp('2011-01-01 10:00'), Timestamp('2011-01-03 10:00')], index=[0, 2], dtype=f'M8[{unit}]')
    tm.assert_series_equal(result, expected)
    idx = DatetimeIndex(['2011-01-01 10:00', NaT, '2011-01-03 10:00', NaT], tz='Asia/Tokyo').as_unit(unit)
    ser = Series(idx)
    assert ser.dtype == f'datetime64[{unit}, Asia/Tokyo]'
    result = ser.dropna()
    expected = Series([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'), Timestamp('2011-01-03 10:00', tz='Asia/Tokyo')], index=[0, 2], dtype=f'datetime64[{unit}, Asia/Tokyo]')
    assert result.dtype == f'datetime64[{unit}, Asia/Tokyo]'
    tm.assert_series_equal(result, expected)