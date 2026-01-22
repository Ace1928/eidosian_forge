import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_string_index(self):
    df = DataFrame([['A', 'group_1', Timestamp(2019, 1, 1, 9)], ['B', 'group_1', Timestamp(2019, 1, 2, 9)], ['Z', 'group_2', Timestamp(2019, 1, 3, 9)], ['H', 'group_1', Timestamp(2019, 1, 6, 9)], ['E', 'group_2', Timestamp(2019, 1, 20, 9)]], columns=['index', 'group', 'eventTime']).set_index('index')
    groups = df.groupby('group')
    df['count_to_date'] = groups.cumcount()
    rolling_groups = groups.rolling('10d', on='eventTime')
    result = rolling_groups.apply(lambda df: df.shape[0])
    expected = DataFrame([['A', 'group_1', Timestamp(2019, 1, 1, 9), 1.0], ['B', 'group_1', Timestamp(2019, 1, 2, 9), 2.0], ['H', 'group_1', Timestamp(2019, 1, 6, 9), 3.0], ['Z', 'group_2', Timestamp(2019, 1, 3, 9), 1.0], ['E', 'group_2', Timestamp(2019, 1, 20, 9), 1.0]], columns=['index', 'group', 'eventTime', 'count_to_date']).set_index(['group', 'index'])
    tm.assert_frame_equal(result, expected)