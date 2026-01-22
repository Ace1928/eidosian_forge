from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('closed,expected', [('left', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 18, 21]), ('neither', [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 13, 8]), ('right', [0, 1, 3, 6, 10, 15, 21, 28, 21, 17]), ('both', [0, 1, 3, 6, 10, 15, 21, 28, 26, 30])])
def test_variable_offset_window_nonunique(closed, expected, frame_or_series):
    index = DatetimeIndex(['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-02', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-04', '2011-01-05', '2011-01-06'])
    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    offset = BusinessDay(2)
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result = df.rolling(indexer, closed=closed, min_periods=1).sum()
    tm.assert_equal(result, expected)