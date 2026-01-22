import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_assert_series_equal_extension_dtype_mismatch():
    left = Series(pd.array([1, 2, 3], dtype='Int64'))
    right = left.astype(int)
    msg = 'Attributes of Series are different\n\nAttribute "dtype" are different\n\\[left\\]:  Int64\n\\[right\\]: int[32|64]'
    tm.assert_series_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)