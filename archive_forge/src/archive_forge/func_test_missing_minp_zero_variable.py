from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_missing_minp_zero_variable():
    x = Series([np.nan] * 4, index=DatetimeIndex(['2017-01-01', '2017-01-04', '2017-01-06', '2017-01-07']))
    result = x.rolling(Timedelta('2d'), min_periods=0).sum()
    expected = Series(0.0, index=x.index)
    tm.assert_series_equal(result, expected)