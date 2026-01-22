import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values, expected', [(['1', '2', None], Series([1, 2, np.nan], dtype='Int64')), (['1', '2', '3'], Series([1, 2, 3], dtype='Int64')), (['1', '2', 3], Series([1, 2, 3], dtype='Int64')), (['1', '2', 3.5], Series([1, 2, 3.5], dtype='Float64')), (['1', None, 3.5], Series([1, np.nan, 3.5], dtype='Float64')), (['1', '2', '3.5'], Series([1, 2, 3.5], dtype='Float64'))])
def test_to_numeric_from_nullable_string(values, nullable_string_dtype, expected):
    s = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(s)
    tm.assert_series_equal(result, expected)