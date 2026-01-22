import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_iloc_dateoffset_days(self):
    df = DataFrame(list(range(10)), index=date_range('01-01-2022', periods=10, freq=DateOffset(days=1)))
    result = df.loc['2022-01-01':'2022-01-03']
    expected = DataFrame([0, 1, 2], index=DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', freq=DateOffset(days=1)))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(list(range(10)), index=date_range('01-01-2022', periods=10, freq=DateOffset(days=1, hours=2)))
    result = df.loc['2022-01-01':'2022-01-03']
    expected = DataFrame([0, 1, 2], index=DatetimeIndex(['2022-01-01 00:00:00', '2022-01-02 02:00:00', '2022-01-03 04:00:00'], dtype='datetime64[ns]', freq=DateOffset(days=1, hours=2)))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(list(range(10)), index=date_range('01-01-2022', periods=10, freq=DateOffset(minutes=3)))
    result = df.loc['2022-01-01':'2022-01-03']
    tm.assert_frame_equal(result, df)