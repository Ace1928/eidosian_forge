import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
@td.skip_if_no('pyarrow')
def test_log_arrow_backed_missing_value():
    ser = Series([1, 2, None], dtype='float64[pyarrow]')
    result = np.log(ser)
    expected = np.log(Series([1, 2, None], dtype='float64'))
    tm.assert_series_equal(result, expected)