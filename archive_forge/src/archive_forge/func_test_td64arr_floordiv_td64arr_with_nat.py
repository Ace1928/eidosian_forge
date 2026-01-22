from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_floordiv_td64arr_with_nat(self, box_with_array, using_array_manager):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    left = Series([1000, 222330, 30], dtype='timedelta64[ns]')
    right = Series([1000, 222330, None], dtype='timedelta64[ns]')
    left = tm.box_expected(left, box)
    right = tm.box_expected(right, box)
    expected = np.array([1.0, 1.0, np.nan], dtype=np.float64)
    expected = tm.box_expected(expected, xbox)
    if box is DataFrame and using_array_manager:
        expected[[0, 1]] = expected[[0, 1]].astype('int64')
    with tm.maybe_produces_warning(RuntimeWarning, box is pd.array, check_stacklevel=False):
        result = left // right
    tm.assert_equal(result, expected)
    with tm.maybe_produces_warning(RuntimeWarning, box is pd.array, check_stacklevel=False):
        result = np.asarray(left) // right
    tm.assert_equal(result, expected)