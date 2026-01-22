import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_pad():
    s = Series(np.arange(10), dtype='int64')
    s2 = s[::2]
    reindexed = s2.reindex(s.index, method='pad')
    reindexed2 = s2.reindex(s.index, method='ffill')
    tm.assert_series_equal(reindexed, reindexed2)
    expected = Series([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    tm.assert_series_equal(reindexed, expected)