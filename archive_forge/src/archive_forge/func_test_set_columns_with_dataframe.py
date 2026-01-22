import numpy as np
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_columns_with_dataframe(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = DataFrame({'c': [7, 8, 9], 'd': [10, 11, 12]})
    df[['c', 'd']] = df2
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
    else:
        assert not np.shares_memory(get_array(df, 'c'), get_array(df2, 'c'))
    df2.iloc[0, 0] = 0
    tm.assert_series_equal(df['c'], Series([7, 8, 9], name='c'))