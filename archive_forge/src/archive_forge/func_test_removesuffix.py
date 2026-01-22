from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('suffix, expected', [('c', ['ab', 'a b ', 'b']), ('bc', ['ab', 'a b c', ''])])
def test_removesuffix(any_string_dtype, suffix, expected):
    ser = Series(['ab', 'a b c', 'bc'], dtype=any_string_dtype)
    result = ser.str.removesuffix(suffix)
    ser_expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)