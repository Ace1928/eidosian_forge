from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_int_dtype_with_bool(self):
    s_0123 = Series(range(4), dtype='int64')
    expected = Series([False] * 4)
    result = s_0123 & False
    tm.assert_series_equal(result, expected)
    warn_msg = 'Logical ops \\(and, or, xor\\) between Pandas objects and dtype-less sequences'
    with tm.assert_produces_warning(FutureWarning, match=warn_msg):
        result = s_0123 & [False]
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=warn_msg):
        result = s_0123 & (False,)
    tm.assert_series_equal(result, expected)
    result = s_0123 ^ False
    expected = Series([False, True, True, True])
    tm.assert_series_equal(result, expected)