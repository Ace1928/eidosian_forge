import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewma_nan_handling():
    s = Series([1.0] + [np.nan] * 5 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([1.0] * len(s)))
    s = Series([np.nan] * 2 + [1.0] + [np.nan] * 2 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([np.nan] * 2 + [1.0] * 4))