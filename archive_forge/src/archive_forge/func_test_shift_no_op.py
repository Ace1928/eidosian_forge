import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_shift_no_op(using_copy_on_write):
    df = DataFrame([[1, 2], [3, 4], [5, 6]], index=date_range('2020-01-01', '2020-01-03'), columns=['a', 'b'])
    df_orig = df.copy()
    df2 = df.shift(periods=0)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
        assert np.shares_memory(get_array(df, 'b'), get_array(df2, 'b'))
    tm.assert_frame_equal(df2, df_orig)