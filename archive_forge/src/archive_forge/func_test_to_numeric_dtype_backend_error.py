import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_numeric_dtype_backend_error(dtype_backend):
    ser = Series(['a', 'b', ''])
    expected = ser.copy()
    with pytest.raises(ValueError, match='Unable to parse string'):
        to_numeric(ser, dtype_backend=dtype_backend)
    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_numeric(ser, dtype_backend=dtype_backend, errors='ignore')
    tm.assert_series_equal(result, expected)
    result = to_numeric(ser, dtype_backend=dtype_backend, errors='coerce')
    if dtype_backend == 'pyarrow':
        dtype = 'double[pyarrow]'
    else:
        dtype = 'Float64'
    expected = Series([np.nan, np.nan, np.nan], dtype=dtype)
    tm.assert_series_equal(result, expected)