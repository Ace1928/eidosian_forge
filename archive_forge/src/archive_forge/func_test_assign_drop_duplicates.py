import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', ['assign', 'drop_duplicates'])
def test_assign_drop_duplicates(using_copy_on_write, method):
    df = DataFrame({'a': [1, 2, 3]})
    df_orig = df.copy()
    df2 = getattr(df, method)()
    df2._mgr._verify_integrity()
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)