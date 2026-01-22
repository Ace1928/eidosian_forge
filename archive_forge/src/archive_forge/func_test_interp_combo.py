import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_combo(self):
    df = DataFrame({'A': [1.0, 2.0, np.nan, 4.0], 'B': [1, 4, 9, np.nan], 'C': [1, 2, 3, 5], 'D': list('abcd')})
    result = df['A'].interpolate()
    expected = Series([1.0, 2.0, 3.0, 4.0], name='A')
    tm.assert_series_equal(result, expected)
    msg = "The 'downcast' keyword in Series.interpolate is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df['A'].interpolate(downcast='infer')
    expected = Series([1, 2, 3, 4], name='A')
    tm.assert_series_equal(result, expected)