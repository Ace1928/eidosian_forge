import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_midx_slice(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3], index=pd.MultiIndex.from_arrays([[1, 1, 2], [3, 4, 5]]))
    ser_orig = ser.copy()
    result = ser[1]
    assert np.shares_memory(get_array(ser), get_array(result))
    with tm.assert_cow_warning(warn_copy_on_write):
        result.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3], index=pd.MultiIndex.from_arrays([[1, 1, 2], [3, 4, 5]]))
        tm.assert_series_equal(ser, expected)