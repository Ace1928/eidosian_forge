import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_with_columns_and_limit(self):
    df = DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]], columns=list('ABCD'))
    result = df.fillna(axis=1, value=100, limit=1)
    result2 = df.fillna(axis=1, value=100, limit=2)
    expected = DataFrame({'A': Series([100, 3, 100, 100], dtype='float64'), 'B': [2, 4, np.nan, 3], 'C': [np.nan, 100, np.nan, np.nan], 'D': Series([0, 1, 5, 4], dtype='float64')}, index=[0, 1, 2, 3])
    expected2 = DataFrame({'A': Series([100, 3, 100, 100], dtype='float64'), 'B': Series([2, 4, 100, 3], dtype='float64'), 'C': [100, 100, np.nan, 100], 'D': Series([0, 1, 5, 4], dtype='float64')}, index=[0, 1, 2, 3])
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(result2, expected2)