import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_coerce_single_column(using_copy_on_write, using_array_manager):
    df = DataFrame({'a': [1.5, 2, 3], 'b': 100.5})
    df_orig = df.copy()
    df2 = df.replace(to_replace=1.5, value='a')
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    elif not using_array_manager:
        assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    if using_copy_on_write:
        df2.loc[0, 'b'] = 0.5
        tm.assert_frame_equal(df, df_orig)
        assert not np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))