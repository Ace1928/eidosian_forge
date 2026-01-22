import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_ea_noop_shares_memory(using_copy_on_write, any_numeric_ea_and_arrow_dtype):
    df = DataFrame({'a': [1, NA, 3], 'b': 1}, dtype=any_numeric_ea_and_arrow_dtype)
    df_orig = df.copy()
    df2 = df.fillna(100)
    assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
        assert not df2._mgr._has_no_reference(1)
    elif isinstance(df.dtypes.iloc[0], ArrowDtype):
        assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
    else:
        assert not np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
    tm.assert_frame_equal(df_orig, df)
    df2.iloc[0, 1] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
        assert df2._mgr._has_no_reference(1)
        assert df._mgr._has_no_reference(1)
    tm.assert_frame_equal(df_orig, df)