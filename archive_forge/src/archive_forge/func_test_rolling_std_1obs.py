from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_std_1obs():
    vals = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = vals.rolling(1, min_periods=1).std()
    expected = Series([np.nan] * 5)
    tm.assert_series_equal(result, expected)
    result = vals.rolling(1, min_periods=1).std(ddof=0)
    expected = Series([0.0] * 5)
    tm.assert_series_equal(result, expected)
    result = Series([np.nan, np.nan, 3, 4, 5]).rolling(3, min_periods=2).std()
    assert np.isnan(result[2])