import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_replace_columnwise_no_op_inplace(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
    view = df[:]
    df_orig = df.copy()
    df.replace({'a': 10}, 100, inplace=True)
    if using_copy_on_write:
        assert np.shares_memory(get_array(view, 'a'), get_array(df, 'a'))
        df.iloc[0, 0] = 100
        tm.assert_frame_equal(view, df_orig)