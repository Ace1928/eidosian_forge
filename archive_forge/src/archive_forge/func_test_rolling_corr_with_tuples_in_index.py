import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_rolling_corr_with_tuples_in_index():
    df = DataFrame({'a': [(1, 2), (1, 2), (1, 2)], 'b': [4, 5, 6]})
    gb = df.groupby(['a'])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples([((1, 2), 0), ((1, 2), 1), ((1, 2), 2)], names=['a', None])
    expected = DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [np.nan, 1.0, 1.0]}, index=index)
    tm.assert_frame_equal(result, expected)