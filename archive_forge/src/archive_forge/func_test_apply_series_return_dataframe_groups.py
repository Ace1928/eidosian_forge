from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_series_return_dataframe_groups():
    tdf = DataFrame({'day': {0: pd.Timestamp('2015-02-24 00:00:00'), 1: pd.Timestamp('2015-02-24 00:00:00'), 2: pd.Timestamp('2015-02-24 00:00:00'), 3: pd.Timestamp('2015-02-24 00:00:00'), 4: pd.Timestamp('2015-02-24 00:00:00')}, 'userAgent': {0: 'some UA string', 1: 'some UA string', 2: 'some UA string', 3: 'another UA string', 4: 'some UA string'}, 'userId': {0: '17661101', 1: '17661101', 2: '17661101', 3: '17661101', 4: '17661101'}})

    def most_common_values(df):
        return Series({c: s.value_counts().index[0] for c, s in df.items()})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = tdf.groupby('day').apply(most_common_values)['userId']
    expected = Series(['17661101'], index=pd.DatetimeIndex(['2015-02-24'], name='day'), name='userId')
    tm.assert_series_equal(result, expected)