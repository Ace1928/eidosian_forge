import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_extension_dtype_mismatch():
    left = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    right = left.astype(int)
    msg = 'Attributes of DataFrame\\.iloc\\[:, 0\\] \\(column name="a"\\) are different\n\nAttribute "dtype" are different\n\\[left\\]:  Int64\n\\[right\\]: int[32|64]'
    tm.assert_frame_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)