from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_selection_methods_of_assigned_col():
    df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = DataFrame(data={'c': [7, 8, 9]}, index=[2, 1, 0])
    df['c'] = df2['c']
    df.at[1, 'c'] = 11
    result = df
    expected = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [9, 11, 7]})
    tm.assert_frame_equal(result, expected)
    result = df.at[1, 'c']
    assert result == 11
    result = df['c']
    expected = Series([9, 11, 7], name='c')
    tm.assert_series_equal(result, expected)
    result = df[['c']]
    expected = DataFrame({'c': [9, 11, 7]})
    tm.assert_frame_equal(result, expected)