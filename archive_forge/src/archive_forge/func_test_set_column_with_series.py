import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_column_with_series(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    ser = Series([1, 2, 3])
    df['c'] = ser
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'c'), get_array(ser))
    else:
        assert not np.shares_memory(get_array(df, 'c'), get_array(ser))
    ser.iloc[0] = 0
    assert ser.iloc[0] == 0
    tm.assert_series_equal(df['c'], Series([1, 2, 3], name='c'))