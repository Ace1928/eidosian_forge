import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_listlike(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
    df_orig = df.copy()
    result = df.replace([200, 201], [11, 11])
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df)
    result = df.replace([200, 2], [10, 10])
    assert not np.shares_memory(get_array(df, 'a'), get_array(result, 'a'))
    tm.assert_frame_equal(df, df_orig)