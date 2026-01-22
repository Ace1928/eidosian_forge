import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_empty_list(using_copy_on_write):
    df = DataFrame({'a': [1, 2]})
    df2 = df.replace([], [])
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
        assert not df._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    arr_a = get_array(df, 'a')
    df.replace([], [])
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'a'), arr_a)
        assert not df._mgr._has_no_reference(0)
        assert not df2._mgr._has_no_reference(0)