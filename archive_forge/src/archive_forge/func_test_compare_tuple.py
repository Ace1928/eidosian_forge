import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_tuple():
    mi = MultiIndex.from_product([[1, 2]] * 2)
    all_false = np.array([False, False, False, False])
    result = mi == mi[0]
    expected = np.array([True, False, False, False])
    tm.assert_numpy_array_equal(result, expected)
    result = mi != mi[0]
    tm.assert_numpy_array_equal(result, ~expected)
    result = mi < mi[0]
    tm.assert_numpy_array_equal(result, all_false)
    result = mi <= mi[0]
    tm.assert_numpy_array_equal(result, expected)
    result = mi > mi[0]
    tm.assert_numpy_array_equal(result, ~expected)
    result = mi >= mi[0]
    tm.assert_numpy_array_equal(result, ~all_false)