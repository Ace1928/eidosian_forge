import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
def test_to_numpy_na_value_with_nan():
    arr = FloatingArray(np.array([0.0, np.nan, 0.0]), np.array([False, False, True]))
    result = arr.to_numpy(dtype='float64', na_value=-1)
    expected = np.array([0.0, np.nan, -1.0], dtype='float64')
    tm.assert_numpy_array_equal(result, expected)