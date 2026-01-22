from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_axis_one_with_nan():
    df = DataFrame([[0, 1, 2, 4, np.nan, np.nan, np.nan], [0, 1, 2, np.nan, np.nan, np.nan, np.nan], [0, 2, 2, np.nan, 2, np.nan, 1]])
    msg = 'Support for axis=1 in DataFrame.rolling is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=7, min_periods=1, axis='columns').sum()
    expected = DataFrame([[0.0, 1.0, 3.0, 7.0, 7.0, 7.0, 7.0], [0.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0], [0.0, 2.0, 4.0, 4.0, 6.0, 6.0, 7.0]])
    tm.assert_frame_equal(result, expected)