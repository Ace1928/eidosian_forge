import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
def test_frame_getitem_nan_multiindex(nulls_fixture):
    n = nulls_fixture
    cols = ['a', 'b', 'c']
    df = DataFrame([[11, n, 13], [21, n, 23], [31, n, 33], [41, n, 43]], columns=cols).set_index(['a', 'b'])
    df['c'] = df['c'].astype('int64')
    idx = (21, n)
    result = df.loc[:idx]
    expected = DataFrame([[11, n, 13], [21, n, 23]], columns=cols).set_index(['a', 'b'])
    expected['c'] = expected['c'].astype('int64')
    tm.assert_frame_equal(result, expected)
    result = df.loc[idx:]
    expected = DataFrame([[21, n, 23], [31, n, 33], [41, n, 43]], columns=cols).set_index(['a', 'b'])
    expected['c'] = expected['c'].astype('int64')
    tm.assert_frame_equal(result, expected)
    idx1, idx2 = ((21, n), (31, n))
    result = df.loc[idx1:idx2]
    expected = DataFrame([[21, n, 23], [31, n, 33]], columns=cols).set_index(['a', 'b'])
    expected['c'] = expected['c'].astype('int64')
    tm.assert_frame_equal(result, expected)