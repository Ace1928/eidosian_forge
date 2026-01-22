import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_resulting_multiindex(self):
    df = DataFrame({'a': np.arange(8.0), 'b': [1, 2] * 4})
    result = df.groupby('b').rolling(3).mean()
    expected_index = MultiIndex.from_tuples([(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)], names=['b', None])
    tm.assert_index_equal(result.index, expected_index)