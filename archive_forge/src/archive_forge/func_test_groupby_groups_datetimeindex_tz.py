from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_groups_datetimeindex_tz(self):
    dates = ['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00', '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00']
    df = DataFrame({'label': ['a', 'a', 'a', 'b', 'b', 'b'], 'datetime': dates, 'value1': np.arange(6, dtype='int64'), 'value2': [1, 2] * 3})
    df['datetime'] = df['datetime'].apply(lambda d: Timestamp(d, tz='US/Pacific'))
    exp_idx1 = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00', '2011-07-19 09:00:00'], tz='US/Pacific', name='datetime')
    exp_idx2 = Index(['a', 'b'] * 3, name='label')
    exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
    expected = DataFrame({'value1': [0, 3, 1, 4, 2, 5], 'value2': [1, 2, 2, 1, 1, 2]}, index=exp_idx, columns=['value1', 'value2'])
    result = df.groupby(['datetime', 'label']).sum()
    tm.assert_frame_equal(result, expected)
    didx = DatetimeIndex(dates, tz='Asia/Tokyo')
    df = DataFrame({'value1': np.arange(6, dtype='int64'), 'value2': [1, 2, 3, 1, 2, 3]}, index=didx)
    exp_idx = DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], tz='Asia/Tokyo')
    expected = DataFrame({'value1': [3, 5, 7], 'value2': [2, 4, 6]}, index=exp_idx, columns=['value1', 'value2'])
    result = df.groupby(level=0).sum()
    tm.assert_frame_equal(result, expected)