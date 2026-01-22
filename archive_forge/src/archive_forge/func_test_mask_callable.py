import numpy as np
from pandas import (
import pandas._testing as tm
def test_mask_callable(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = df.mask(lambda x: x > 4, lambda x: x + 1)
    exp = DataFrame([[1, 2, 3], [4, 6, 7], [8, 9, 10]])
    tm.assert_frame_equal(result, exp)
    tm.assert_frame_equal(result, df.mask(df > 4, df + 1))
    result = df.mask(lambda x: (x % 2 == 0).values, lambda x: 99)
    exp = DataFrame([[1, 99, 3], [99, 5, 99], [7, 99, 9]])
    tm.assert_frame_equal(result, exp)
    tm.assert_frame_equal(result, df.mask(df % 2 == 0, 99))
    result = (df + 2).mask(lambda x: x > 8, lambda x: x + 10)
    exp = DataFrame([[3, 4, 5], [6, 7, 8], [19, 20, 21]])
    tm.assert_frame_equal(result, exp)
    tm.assert_frame_equal(result, (df + 2).mask(df + 2 > 8, df + 2 + 10))