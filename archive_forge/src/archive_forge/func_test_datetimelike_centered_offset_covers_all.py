from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window,closed,expected', [('3s', 'right', [3.0, 3.0, 3.0]), ('3s', 'both', [3.0, 3.0, 3.0]), ('3s', 'left', [3.0, 3.0, 3.0]), ('3s', 'neither', [3.0, 3.0, 3.0]), ('2s', 'right', [3.0, 2.0, 2.0]), ('2s', 'both', [3.0, 3.0, 3.0]), ('2s', 'left', [1.0, 3.0, 3.0]), ('2s', 'neither', [1.0, 2.0, 2.0])])
def test_datetimelike_centered_offset_covers_all(window, closed, expected, frame_or_series):
    index = [Timestamp('20130101 09:00:01'), Timestamp('20130101 09:00:02'), Timestamp('20130101 09:00:02')]
    df = frame_or_series([1, 1, 1], index=index)
    result = df.rolling(window, closed=closed, center=True).sum()
    expected = frame_or_series(expected, index=index)
    tm.assert_equal(result, expected)