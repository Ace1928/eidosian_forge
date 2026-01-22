import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_count_closed_on(self, unit):
    df = DataFrame({'column1': range(6), 'column2': range(6), 'group': 3 * ['A', 'B'], 'date': date_range(end='20190101', periods=6, unit=unit)})
    result = df.groupby('group').rolling('3d', on='date', closed='left')['column1'].count()
    dti = DatetimeIndex(['2018-12-27', '2018-12-29', '2018-12-31', '2018-12-28', '2018-12-30', '2019-01-01'], dtype=f'M8[{unit}]')
    mi = MultiIndex.from_arrays([['A', 'A', 'A', 'B', 'B', 'B'], dti], names=['group', 'date'])
    expected = Series([np.nan, 1.0, 1.0, np.nan, 1.0, 1.0], name='column1', index=mi)
    tm.assert_series_equal(result, expected)