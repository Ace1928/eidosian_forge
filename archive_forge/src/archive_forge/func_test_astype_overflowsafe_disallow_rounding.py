import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
import pandas._testing as tm
def test_astype_overflowsafe_disallow_rounding(self):
    arr = np.array([-1500, 1500], dtype='M8[ns]')
    dtype = np.dtype('M8[us]')
    msg = "Cannot losslessly cast '-1500 ns' to us"
    with pytest.raises(ValueError, match=msg):
        astype_overflowsafe(arr, dtype, round_ok=False)
    result = astype_overflowsafe(arr, dtype, round_ok=True)
    expected = arr.astype(dtype)
    tm.assert_numpy_array_equal(result, expected)