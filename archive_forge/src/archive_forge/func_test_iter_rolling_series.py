from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('ser,expected,window, min_periods', [(Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3, None), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3, 1), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])], 2, 1), (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])], 2, 2), (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 0), (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 1), (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2, 0), (Series([], dtype='int64'), [], 2, 1)])
def test_iter_rolling_series(ser, expected, window, min_periods):
    expected = [Series(values, index=index) for values, index in expected]
    for expected, actual in zip(expected, ser.rolling(window, min_periods=min_periods)):
        tm.assert_series_equal(actual, expected)