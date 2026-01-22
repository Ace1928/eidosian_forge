import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_diff_dt64(self):
    ser = Series(date_range('20130102', periods=5))
    result = ser.diff()
    expected = ser - ser.shift(1)
    tm.assert_series_equal(result, expected)
    result = result - result.shift(1)
    expected = expected.diff()
    tm.assert_series_equal(result, expected)