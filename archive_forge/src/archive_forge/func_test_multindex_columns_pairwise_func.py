import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_multindex_columns_pairwise_func(self):
    columns = MultiIndex.from_arrays([['M', 'N'], ['P', 'Q']], names=['a', 'b'])
    df = DataFrame(np.ones((5, 2)), columns=columns)
    result = df.rolling(3).corr()
    expected = DataFrame(np.nan, index=MultiIndex.from_arrays([np.repeat(np.arange(5, dtype=np.int64), 2), ['M', 'N'] * 5, ['P', 'Q'] * 5], names=[None, 'a', 'b']), columns=columns)
    tm.assert_frame_equal(result, expected)