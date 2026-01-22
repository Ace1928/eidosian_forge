import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_list_none_inplace_refs(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': ['a', 'b', 'c']})
    arr = get_array(df, 'a')
    df_orig = df.copy()
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace(['a'], value=None, inplace=True)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
        assert not np.shares_memory(arr, get_array(df, 'a'))
        tm.assert_frame_equal(df_orig, view)
    else:
        assert np.shares_memory(arr, get_array(df, 'a'))