from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('method, exp', [['index', [1, 1, 0]], ['rindex', [3, 1, 2]]])
def test_index_missing(any_string_dtype, method, exp):
    ser = Series(['abcb', 'ab', 'bcbe', np.nan], dtype=any_string_dtype)
    expected_dtype = np.float64 if any_string_dtype in object_pyarrow_numpy else 'Int64'
    result = getattr(ser.str, method)('b')
    expected = Series(exp + [np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)