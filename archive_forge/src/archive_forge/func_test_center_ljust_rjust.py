from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_center_ljust_rjust(any_string_dtype):
    s = Series(['a', 'b', np.nan, 'c', np.nan, 'eeeeee'], dtype=any_string_dtype)
    result = s.str.center(5)
    expected = Series(['  a  ', '  b  ', np.nan, '  c  ', np.nan, 'eeeeee'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.ljust(5)
    expected = Series(['a    ', 'b    ', np.nan, 'c    ', np.nan, 'eeeeee'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result = s.str.rjust(5)
    expected = Series(['    a', '    b', np.nan, '    c', np.nan, 'eeeeee'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)