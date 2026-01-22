from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_string_slice_get_syntax(any_string_dtype):
    ser = Series(['YYY', 'B', 'C', 'YYYYYYbYYY', 'BYYYcYYY', np.nan, 'CYYYBYYY', 'dog', 'cYYYt'], dtype=any_string_dtype)
    result = ser.str[0]
    expected = ser.str.get(0)
    tm.assert_series_equal(result, expected)
    result = ser.str[:3]
    expected = ser.str.slice(stop=3)
    tm.assert_series_equal(result, expected)
    result = ser.str[2::-1]
    expected = ser.str.slice(start=2, step=-1)
    tm.assert_series_equal(result, expected)