import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_ignore_all_good(self):
    df = DataFrame({'A': [1, 2, np.nan, 4], 'B': [1, 2, 3, 4], 'C': [1.0, 2.0, np.nan, 4.0], 'D': [1.0, 2.0, 3.0, 4.0]})
    expected = DataFrame({'A': np.array([1, 2, 3, 4], dtype='float64'), 'B': np.array([1, 2, 3, 4], dtype='int64'), 'C': np.array([1.0, 2.0, 3, 4.0], dtype='float64'), 'D': np.array([1.0, 2.0, 3.0, 4.0], dtype='float64')})
    msg = "The 'downcast' keyword in DataFrame.interpolate is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.interpolate(downcast=None)
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df[['B', 'D']].interpolate(downcast=None)
    tm.assert_frame_equal(result, df[['B', 'D']])