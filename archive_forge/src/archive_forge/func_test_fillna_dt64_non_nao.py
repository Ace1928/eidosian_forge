from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_dt64_non_nao(self):
    ser = Series([Timestamp('2010-01-01'), NaT, Timestamp('2000-01-01')])
    val = np.datetime64('1975-04-05', 'ms')
    result = ser.fillna(val)
    expected = Series([Timestamp('2010-01-01'), Timestamp('1975-04-05'), Timestamp('2000-01-01')])
    tm.assert_series_equal(result, expected)