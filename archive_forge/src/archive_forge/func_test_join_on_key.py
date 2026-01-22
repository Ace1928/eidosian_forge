import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_join_on_key(using_copy_on_write):
    df_index = Index(['a', 'b', 'c'], name='key')
    df1 = DataFrame({'a': [1, 2, 3]}, index=df_index.copy(deep=True))
    df2 = DataFrame({'b': [4, 5, 6]}, index=df_index.copy(deep=True))
    df1_orig = df1.copy()
    df2_orig = df2.copy()
    result = df1.join(df2, on='key')
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert np.shares_memory(get_array(result, 'b'), get_array(df2, 'b'))
        assert np.shares_memory(get_array(result.index), get_array(df1.index))
        assert not np.shares_memory(get_array(result.index), get_array(df2.index))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert not np.shares_memory(get_array(result, 'b'), get_array(df2, 'b'))
    result.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert np.shares_memory(get_array(result, 'b'), get_array(df2, 'b'))
    result.iloc[0, 1] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'b'), get_array(df2, 'b'))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)