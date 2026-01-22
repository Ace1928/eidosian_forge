from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_dont_cast_strings(self):
    vals = ['0', '1.5', '-0.3']
    for val in vals:
        ser = Series([0, 1, np.nan, np.nan, 4], dtype='float64')
        result = ser.fillna(val)
        expected = Series([0, 1, val, val, 4], dtype='object')
        tm.assert_series_equal(result, expected)