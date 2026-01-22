from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_ops_bool_dtype_with_ndarray(self):
    left = Series([True, True, True, False, True])
    right = [True, False, None, True, np.nan]
    msg = 'Logical ops \\(and, or, xor\\) between Pandas objects and dtype-less sequences'
    expected = Series([True, False, False, False, False])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = left & right
    tm.assert_series_equal(result, expected)
    result = left & np.array(right)
    tm.assert_series_equal(result, expected)
    result = left & Index(right)
    tm.assert_series_equal(result, expected)
    result = left & Series(right)
    tm.assert_series_equal(result, expected)
    expected = Series([True, True, True, True, True])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = left | right
    tm.assert_series_equal(result, expected)
    result = left | np.array(right)
    tm.assert_series_equal(result, expected)
    result = left | Index(right)
    tm.assert_series_equal(result, expected)
    result = left | Series(right)
    tm.assert_series_equal(result, expected)
    expected = Series([False, True, True, True, True])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = left ^ right
    tm.assert_series_equal(result, expected)
    result = left ^ np.array(right)
    tm.assert_series_equal(result, expected)
    result = left ^ Index(right)
    tm.assert_series_equal(result, expected)
    result = left ^ Series(right)
    tm.assert_series_equal(result, expected)