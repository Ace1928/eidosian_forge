import re
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nanosecond_getitem_setitem_with_tz(self):
    data = ['2016-06-28 08:30:00.123456789']
    index = pd.DatetimeIndex(data, dtype='datetime64[ns, America/Chicago]')
    df = DataFrame({'a': [10]}, index=index)
    result = df.loc[df.index[0]]
    expected = Series(10, index=['a'], name=df.index[0])
    tm.assert_series_equal(result, expected)
    result = df.copy()
    result.loc[df.index[0], 'a'] = -1
    expected = DataFrame(-1, index=index, columns=['a'])
    tm.assert_frame_equal(result, expected)