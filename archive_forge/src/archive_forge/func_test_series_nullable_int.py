import numpy as np
import pytest
from pandas.errors import DataError
from pandas.core.dtypes.common import pandas_dtype
from pandas import (
import pandas._testing as tm
def test_series_nullable_int(any_signed_int_ea_dtype, step):
    ser = Series([0, 1, NA], dtype=any_signed_int_ea_dtype)
    result = ser.rolling(2, step=step).mean()
    expected = Series([np.nan, 0.5, np.nan])[::step]
    tm.assert_series_equal(result, expected)