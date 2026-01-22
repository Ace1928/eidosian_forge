import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_if_no('pyarrow')
def test_to_numpy_arrow_dtype_given():
    ser = Series([1, NA], dtype='int64[pyarrow]')
    result = ser.to_numpy(dtype='float64')
    expected = np.array([1.0, np.nan])
    tm.assert_numpy_array_equal(result, expected)