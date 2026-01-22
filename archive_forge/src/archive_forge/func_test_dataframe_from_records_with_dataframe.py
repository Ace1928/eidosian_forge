import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_dataframe_from_records_with_dataframe(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]})
    df_orig = df.copy()
    with tm.assert_produces_warning(FutureWarning):
        df2 = DataFrame.from_records(df)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
    assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, df2)