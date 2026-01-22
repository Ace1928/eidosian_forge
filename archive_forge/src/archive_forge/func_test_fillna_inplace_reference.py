import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_inplace_reference(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1.5, np.nan], 'b': 1})
    df_orig = df.copy()
    arr_a = get_array(df, 'a')
    arr_b = get_array(df, 'b')
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.fillna(5.5, inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), arr_a)
        assert np.shares_memory(get_array(df, 'b'), arr_b)
        assert view._mgr._has_no_reference(0)
        assert df._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'a'), arr_a)
        assert np.shares_memory(get_array(df, 'b'), arr_b)
    expected = DataFrame({'a': [1.5, 5.5], 'b': 1})
    tm.assert_frame_equal(df, expected)