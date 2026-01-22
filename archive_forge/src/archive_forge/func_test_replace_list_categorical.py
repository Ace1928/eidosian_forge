import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_list_categorical(using_copy_on_write):
    df = DataFrame({'a': ['a', 'b', 'c']}, dtype='category')
    arr = get_array(df, 'a')
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.replace(['c'], value='a', inplace=True)
    assert np.shares_memory(arr.codes, get_array(df, 'a').codes)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
    df_orig = df.copy()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df2 = df.replace(['b'], value='a')
    assert not np.shares_memory(arr.codes, get_array(df2, 'a').codes)
    tm.assert_frame_equal(df, df_orig)