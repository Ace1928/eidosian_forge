import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_limit_direction(self):
    s = Series([1, 3, np.nan, np.nan, np.nan, 11])
    expected = Series([1.0, 3.0, np.nan, 7.0, 9.0, 11.0])
    result = s.interpolate(method='linear', limit=2, limit_direction='backward')
    tm.assert_series_equal(result, expected)
    expected = Series([1.0, 3.0, 5.0, np.nan, 9.0, 11.0])
    result = s.interpolate(method='linear', limit=1, limit_direction='both')
    tm.assert_series_equal(result, expected)
    s = Series([1, 3, np.nan, np.nan, np.nan, 7, 9, np.nan, np.nan, 12, np.nan])
    expected = Series([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0])
    result = s.interpolate(method='linear', limit=2, limit_direction='both')
    tm.assert_series_equal(result, expected)
    expected = Series([1.0, 3.0, 4.0, np.nan, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0])
    result = s.interpolate(method='linear', limit=1, limit_direction='both')
    tm.assert_series_equal(result, expected)