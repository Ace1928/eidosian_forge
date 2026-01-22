import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_between_inclusive_string(self):
    series = Series(date_range('1/1/2000', periods=10))
    left, right = series[[2, 7]]
    result = series.between(left, right, inclusive='both')
    expected = (series >= left) & (series <= right)
    tm.assert_series_equal(result, expected)
    result = series.between(left, right, inclusive='left')
    expected = (series >= left) & (series < right)
    tm.assert_series_equal(result, expected)
    result = series.between(left, right, inclusive='right')
    expected = (series > left) & (series <= right)
    tm.assert_series_equal(result, expected)
    result = series.between(left, right, inclusive='neither')
    expected = (series > left) & (series < right)
    tm.assert_series_equal(result, expected)