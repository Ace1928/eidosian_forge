from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('method, expected', [('isnumeric', [False, np.nan, True, False, np.nan, True, False]), ('isdecimal', [False, np.nan, False, False, np.nan, True, False])])
def test_isnumeric_unicode_missing(method, expected, any_string_dtype):
    values = ['A', np.nan, '¼', '★', np.nan, '３', 'four']
    ser = Series(values, dtype=any_string_dtype)
    expected_dtype = 'object' if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)