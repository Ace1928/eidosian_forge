import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_setitem_series_no_copy_single_block(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()
    df['a'] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(rhs), get_array(df, 'a'))
    df.iloc[0, 0] = 100
    tm.assert_series_equal(rhs, rhs_orig)