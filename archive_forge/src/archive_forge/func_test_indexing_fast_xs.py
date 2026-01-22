import re
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_indexing_fast_xs(self):
    df = DataFrame({'a': date_range('2014-01-01', periods=10, tz='UTC')})
    result = df.iloc[5]
    expected = Series([Timestamp('2014-01-06 00:00:00+0000', tz='UTC')], index=['a'], name=5, dtype='M8[ns, UTC]')
    tm.assert_series_equal(result, expected)
    result = df.loc[5]
    tm.assert_series_equal(result, expected)
    result = df[df.a > df.a[3]]
    expected = df.iloc[4:]
    tm.assert_frame_equal(result, expected)