from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('keys', [['a'], ['a', 'b']])
def test_resample_no_index(keys):
    df = DataFrame([], columns=['a', 'b', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby(keys).resample(rule=pd.to_timedelta('00:00:01')).mean()
    expected = DataFrame(columns=['a', 'b', 'date']).set_index(keys, drop=False)
    expected['date'] = pd.to_datetime(expected['date'])
    expected = expected.set_index('date', append=True, drop=True)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_frame_equal(result, expected)