import numpy as np
from pandas import (
import pandas._testing as tm
def test_unique_uint64(self):
    ser = Series([1, 2, 2 ** 63, 2 ** 63], dtype=np.uint64)
    res = ser.unique()
    exp = np.array([1, 2, 2 ** 63], dtype=np.uint64)
    tm.assert_numpy_array_equal(res, exp)