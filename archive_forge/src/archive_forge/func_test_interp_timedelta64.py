import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_timedelta64(self):
    df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 3]))
    result = df.interpolate(method='time')
    expected = Series([1.0, 2.0, 3.0], index=pd.to_timedelta([1, 2, 3]))
    tm.assert_series_equal(result, expected)
    df = Series([1, np.nan, 3], index=pd.to_timedelta([1, 2, 4]))
    result = df.interpolate(method='time')
    expected = Series([1.0, 1.666667, 3.0], index=pd.to_timedelta([1, 2, 4]))
    tm.assert_series_equal(result, expected)