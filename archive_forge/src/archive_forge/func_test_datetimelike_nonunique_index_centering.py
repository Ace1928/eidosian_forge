from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window,closed,expected', [('2D', 'right', [4, 4, 4, 4, 4, 4, 2, 2]), ('2D', 'left', [2, 2, 4, 4, 4, 4, 4, 4]), ('2D', 'both', [4, 4, 6, 6, 6, 6, 4, 4]), ('2D', 'neither', [2, 2, 2, 2, 2, 2, 2, 2])])
def test_datetimelike_nonunique_index_centering(window, closed, expected, frame_or_series):
    index = DatetimeIndex(['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03', '2020-01-03', '2020-01-04', '2020-01-04'])
    df = frame_or_series([1] * 8, index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling(window, center=True, closed=closed).sum()
    tm.assert_equal(result, expected)