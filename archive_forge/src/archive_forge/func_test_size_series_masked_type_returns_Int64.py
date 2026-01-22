import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
def test_size_series_masked_type_returns_Int64(dtype):
    ser = Series([1, 1, 1], index=['a', 'a', 'b'], dtype=dtype)
    result = ser.groupby(level=0).size()
    expected = Series([2, 1], dtype='Int64', index=['a', 'b'])
    tm.assert_series_equal(result, expected)