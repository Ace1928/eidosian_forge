import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_mask_casts():
    ser = Series([1, 2, 3, 4])
    result = ser.mask(ser > 2, np.nan)
    expected = Series([1, 2, np.nan, np.nan])
    tm.assert_series_equal(result, expected)