import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_setitem_with_array_with_missing(dtype):
    arr = pd.array(['a', 'b', 'c'], dtype=dtype)
    value = np.array(['A', None])
    value_orig = value.copy()
    arr[[0, 1]] = value
    expected = pd.array(['A', pd.NA, 'c'], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)
    tm.assert_numpy_array_equal(value, value_orig)