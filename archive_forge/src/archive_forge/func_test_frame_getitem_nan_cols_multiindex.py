import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
@pytest.mark.parametrize('indexer,expected', [((['b'], ['bar', np.nan]), DataFrame([[2, 3], [5, 6]], columns=MultiIndex.from_tuples([('b', 'bar'), ('b', np.nan)]), dtype='int64')), (['a', 'b'], DataFrame([[1, 2, 3], [4, 5, 6]], columns=MultiIndex.from_tuples([('a', 'foo'), ('b', 'bar'), ('b', np.nan)]), dtype='int64')), (['b'], DataFrame([[2, 3], [5, 6]], columns=MultiIndex.from_tuples([('b', 'bar'), ('b', np.nan)]), dtype='int64')), ((['b'], ['bar']), DataFrame([[2], [5]], columns=MultiIndex.from_tuples([('b', 'bar')]), dtype='int64')), ((['b'], [np.nan]), DataFrame([[3], [6]], columns=MultiIndex(codes=[[1], [-1]], levels=[['a', 'b'], ['bar', 'foo']]), dtype='int64')), (('b', np.nan), Series([3, 6], dtype='int64', name=('b', np.nan)))])
def test_frame_getitem_nan_cols_multiindex(indexer, expected, nulls_fixture):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=MultiIndex.from_tuples([('a', 'foo'), ('b', 'bar'), ('b', nulls_fixture)]), dtype='int64')
    result = df.loc[:, indexer]
    tm.assert_equal(result, expected)