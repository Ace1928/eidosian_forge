from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window', [1, '1d'])
def test_rolling_descending_date_order_with_offset(window, frame_or_series):
    idx = date_range(start='2020-01-01', end='2020-01-03', freq='1d')
    obj = frame_or_series(range(1, 4), index=idx)
    result = obj.rolling('1d', closed='left').sum()
    expected = frame_or_series([np.nan, 1, 2], index=idx)
    tm.assert_equal(result, expected)
    result = obj.iloc[::-1].rolling('1d', closed='left').sum()
    idx = date_range(start='2020-01-03', end='2020-01-01', freq='-1d')
    expected = frame_or_series([np.nan, 3, 2], index=idx)
    tm.assert_equal(result, expected)