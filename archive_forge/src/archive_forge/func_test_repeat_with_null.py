from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('arg, repeat', [[None, 4], ['b', None]])
def test_repeat_with_null(any_string_dtype, arg, repeat):
    ser = Series(['a', arg], dtype=any_string_dtype)
    result = ser.str.repeat([3, repeat])
    expected = Series(['aaa', None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)