import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_diff_bool():
    df = DataFrame({'a': [1, 2, 3, 3, 2], 'b': [True, True, False, False, True]})
    result = df.groupby('a')['b'].diff()
    expected = Series([np.nan, np.nan, np.nan, False, False], name='b')
    tm.assert_series_equal(result, expected)