import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val, dtype, downcast', [(1, 'Int8', 'integer'), (1.5, 'Float32', 'float'), (1, 'Int8', 'signed'), (1, 'int8[pyarrow]', 'integer'), (1.5, 'float[pyarrow]', 'float'), (1, 'int8[pyarrow]', 'signed')])
def test_to_numeric_dtype_backend_downcasting(val, dtype, downcast):
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
        dtype_backend = 'pyarrow'
    else:
        dtype_backend = 'numpy_nullable'
    ser = Series([val, None], dtype=object)
    result = to_numeric(ser, dtype_backend=dtype_backend, downcast=downcast)
    expected = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)