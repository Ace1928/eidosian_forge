import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_between_datetime_object_dtype(self):
    ser = Series(bdate_range('1/1/2000', periods=20), dtype=object)
    ser[::2] = np.nan
    result = ser[ser.between(ser[3], ser[17])]
    expected = ser[3:18].dropna()
    tm.assert_series_equal(result, expected)
    result = ser[ser.between(ser[3], ser[17], inclusive='neither')]
    expected = ser[5:16].dropna()
    tm.assert_series_equal(result, expected)