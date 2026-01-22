from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_parr(self):
    dti = date_range(Timestamp.max - Timedelta(nanoseconds=10), periods=5, freq='ns')
    ser = Series(dti.to_period('ns'))
    ser[2] = NaT
    arr = period_array([Timestamp('2262-04-11 23:47:16.854775797'), Timestamp('2262-04-11 23:47:16.854775798'), Timestamp('2262-04-11 23:47:16.854775798'), Timestamp('2262-04-11 23:47:16.854775800'), Timestamp('2262-04-11 23:47:16.854775801')], freq='ns')
    expected = Series(arr)
    filled = ser.ffill()
    tm.assert_series_equal(filled, expected)