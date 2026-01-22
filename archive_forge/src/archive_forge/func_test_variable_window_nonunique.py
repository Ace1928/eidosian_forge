from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('closed,expected', [('left', [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]), ('neither', [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]), ('right', [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]), ('both', [0, 1, 3, 6, 10, 15, 20, 27, 26, 30])])
def test_variable_window_nonunique(closed, expected, frame_or_series):
    index = DatetimeIndex(['2011-01-01', '2011-01-01', '2011-01-02', '2011-01-02', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-04', '2011-01-05', '2011-01-06'])
    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling('2D', closed=closed).sum()
    tm.assert_equal(result, expected)