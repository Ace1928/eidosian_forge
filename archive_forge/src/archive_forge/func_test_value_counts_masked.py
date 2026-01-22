import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_value_counts_masked(self):
    dtype = 'Int64'
    ser = Series([1, 2, None, 2, None, 3], dtype=dtype)
    result = ser.value_counts(dropna=False)
    expected = Series([2, 2, 1, 1], index=Index([2, None, 1, 3], dtype=dtype), dtype=dtype, name='count')
    tm.assert_series_equal(result, expected)
    result = ser.value_counts(dropna=True)
    expected = Series([2, 1, 1], index=Index([2, 1, 3], dtype=dtype), dtype=dtype, name='count')
    tm.assert_series_equal(result, expected)