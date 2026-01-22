import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_mask_all_false_second_block(using_copy_on_write):
    df = DataFrame({'a': [1.5, 2, 3], 'b': 100.5, 'c': 1, 'd': 2})
    df_orig = df.copy()
    df2 = df.replace(to_replace=1.5, value=55.5)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    else:
        assert not np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    df2.loc[0, 'c'] = 1
    tm.assert_frame_equal(df, df_orig)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))