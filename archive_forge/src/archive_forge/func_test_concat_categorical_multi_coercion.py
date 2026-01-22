import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_multi_coercion(self):
    s1 = Series([1, 3], dtype='category')
    s2 = Series([3, 4], dtype='category')
    s3 = Series([2, 3])
    s4 = Series([2, 2], dtype='category')
    s5 = Series([1, np.nan])
    s6 = Series([1, 3, 2], dtype='category')
    exp = Series([1, 3, 3, 4, 2, 3, 2, 2, 1, np.nan, 1, 3, 2])
    res = pd.concat([s1, s2, s3, s4, s5, s6], ignore_index=True)
    tm.assert_series_equal(res, exp)
    res = s1._append([s2, s3, s4, s5, s6], ignore_index=True)
    tm.assert_series_equal(res, exp)
    exp = Series([1, 3, 2, 1, np.nan, 2, 2, 2, 3, 3, 4, 1, 3])
    res = pd.concat([s6, s5, s4, s3, s2, s1], ignore_index=True)
    tm.assert_series_equal(res, exp)
    res = s6._append([s5, s4, s3, s2, s1], ignore_index=True)
    tm.assert_series_equal(res, exp)