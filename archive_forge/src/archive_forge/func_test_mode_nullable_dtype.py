import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_mode_nullable_dtype(any_numeric_ea_dtype):
    ser = Series([1, 3, 2, pd.NA, 3, 2, pd.NA], dtype=any_numeric_ea_dtype)
    result = ser.mode(dropna=False)
    expected = Series([2, 3, pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)
    ser[-1] = pd.NA
    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.mode(dropna=False)
    expected = Series([pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)