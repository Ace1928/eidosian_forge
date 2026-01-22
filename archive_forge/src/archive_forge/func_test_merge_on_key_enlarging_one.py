import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('func, how', [(lambda df1, df2, **kwargs: merge(df2, df1, on='key', **kwargs), 'right'), (lambda df1, df2, **kwargs: merge(df1, df2, on='key', **kwargs), 'left')])
def test_merge_on_key_enlarging_one(using_copy_on_write, func, how):
    df1 = DataFrame({'key': ['a', 'b', 'c'], 'a': [1, 2, 3]})
    df2 = DataFrame({'key': ['a', 'b'], 'b': [4, 5]})
    df1_orig = df1.copy()
    df2_orig = df2.copy()
    result = func(df1, df2, how=how)
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert not np.shares_memory(get_array(result, 'b'), get_array(df2, 'b'))
        assert df2._mgr._has_no_reference(1)
        assert df2._mgr._has_no_reference(0)
        assert np.shares_memory(get_array(result, 'key'), get_array(df1, 'key')) is (how == 'left')
        assert not np.shares_memory(get_array(result, 'key'), get_array(df2, 'key'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
        assert not np.shares_memory(get_array(result, 'b'), get_array(df2, 'b'))
    if how == 'left':
        result.iloc[0, 1] = 0
    else:
        result.iloc[0, 2] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df1, 'a'))
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)