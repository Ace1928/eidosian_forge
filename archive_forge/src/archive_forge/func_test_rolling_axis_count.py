from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_axis_count(axis_frame):
    df = DataFrame({'x': range(3), 'y': range(3)})
    axis = df._get_axis_number(axis_frame)
    if axis in [0, 'index']:
        msg = "The 'axis' keyword in DataFrame.rolling"
        expected = DataFrame({'x': [1.0, 2.0, 2.0], 'y': [1.0, 2.0, 2.0]})
    else:
        msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
        expected = DataFrame({'x': [1.0, 1.0, 1.0], 'y': [2.0, 2.0, 2.0]})
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(2, axis=axis_frame, min_periods=0).count()
    tm.assert_frame_equal(result, expected)