import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_subset_with_closed(self):
    df = DataFrame({'column1': range(8), 'column2': range(8), 'group': ['A'] * 4 + ['B'] * 4, 'date': [Timestamp(date) for date in ['2019-01-01', '2019-01-01', '2019-01-02', '2019-01-02']] * 2})
    result = df.groupby('group').rolling('1D', on='date', closed='left')['column1'].sum()
    expected = Series([np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0], index=MultiIndex.from_frame(df[['group', 'date']], names=['group', 'date']), name='column1')
    tm.assert_series_equal(result, expected)