from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize(('index', 'window'), [([0, 1, 2, 3, 4], 2), (date_range('2001-01-01', freq='D', periods=5), '2D')])
def test_rolling_corr_timedelta_index(index, window):
    x = Series([1, 2, 3, 4, 5], index=index)
    y = x.copy()
    x.iloc[0:2] = 0.0
    result = x.rolling(window).corr(y)
    expected = Series([np.nan, np.nan, 1, 1, 1], index=index)
    tm.assert_almost_equal(result, expected)