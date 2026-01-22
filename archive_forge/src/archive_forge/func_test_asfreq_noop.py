import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_asfreq_noop(using_copy_on_write):
    df = DataFrame({'a': [0.0, None, 2.0, 3.0]}, index=date_range('1/1/2000', periods=4, freq='min'))
    df_orig = df.copy()
    df2 = df.asfreq(freq='min')
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, 'a'), get_array(df, 'a'))
    tm.assert_frame_equal(df, df_orig)