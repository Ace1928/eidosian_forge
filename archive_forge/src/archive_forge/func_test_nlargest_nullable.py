from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_nlargest_nullable(self, any_numeric_ea_dtype):
    dtype = any_numeric_ea_dtype
    if dtype.startswith('UInt'):
        arr = np.random.default_rng(2).integers(1, 10, 10)
    else:
        arr = np.random.default_rng(2).standard_normal(10)
    arr = arr.astype(dtype.lower(), copy=False)
    ser = Series(arr.copy(), dtype=dtype)
    ser[1] = pd.NA
    result = ser.nlargest(5)
    expected = Series(np.delete(arr, 1), index=ser.index.delete(1)).nlargest(5).astype(dtype)
    tm.assert_series_equal(result, expected)