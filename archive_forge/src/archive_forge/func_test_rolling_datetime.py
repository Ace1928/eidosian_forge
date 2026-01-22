from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_datetime(axis_frame, tz_naive_fixture):
    tz = tz_naive_fixture
    df = DataFrame({i: [1] * 2 for i in date_range('2019-8-01', '2019-08-03', freq='D', tz=tz)})
    if axis_frame in [0, 'index']:
        msg = "The 'axis' keyword in DataFrame.rolling"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.T.rolling('2D', axis=axis_frame).sum().T
    else:
        msg = 'Support for axis=1 in DataFrame.rolling'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.rolling('2D', axis=axis_frame).sum()
    expected = DataFrame({**{i: [1.0] * 2 for i in date_range('2019-8-01', periods=1, freq='D', tz=tz)}, **{i: [2.0] * 2 for i in date_range('2019-8-02', '2019-8-03', freq='D', tz=tz)}})
    tm.assert_frame_equal(result, expected)