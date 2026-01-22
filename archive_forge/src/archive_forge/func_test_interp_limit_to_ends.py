import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_limit_to_ends(self):
    s = Series([np.nan, np.nan, 5, 7, 9, np.nan])
    expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, np.nan])
    result = s.interpolate(method='linear', limit=2, limit_direction='backward')
    tm.assert_series_equal(result, expected)
    expected = Series([5.0, 5.0, 5.0, 7.0, 9.0, 9.0])
    result = s.interpolate(method='linear', limit=2, limit_direction='both')
    tm.assert_series_equal(result, expected)