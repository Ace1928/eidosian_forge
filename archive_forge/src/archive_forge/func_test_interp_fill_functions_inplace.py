import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('func', ['ffill', 'bfill'])
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_interp_fill_functions_inplace(using_copy_on_write, func, warn_copy_on_write, dtype):
    df = DataFrame({'a': [1, np.nan, 2]}, dtype=dtype)
    df_orig = df.copy()
    arr = get_array(df, 'a')
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write and dtype == 'float64'):
        getattr(df, func)(inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(arr, get_array(df, 'a'))
        tm.assert_frame_equal(df_orig, view)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
    else:
        assert np.shares_memory(arr, get_array(df, 'a')) is (dtype == 'float64')