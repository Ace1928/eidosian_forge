import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_array(self):
    index = MultiIndex.from_product([[1, 2, 3], ['A', 'B', 'C']])
    x = Series(index=index, data=range(9), dtype=np.float64)
    y = np.array([1, 3])
    expected = Series(data=[0, 1, 2, 6, 7, 8], index=MultiIndex.from_product([[1, 3], ['A', 'B', 'C']]), dtype=np.float64)
    result = x.loc[y]
    tm.assert_series_equal(result, expected)
    empty = np.array([])
    expected = Series([], index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64), dtype='float64')
    result = x.loc[empty]
    tm.assert_series_equal(result, expected)
    scalar = np.int64(1)
    expected = Series(data=[0, 1, 2], index=['A', 'B', 'C'], dtype=np.float64)
    result = x.loc[scalar]
    tm.assert_series_equal(result, expected)