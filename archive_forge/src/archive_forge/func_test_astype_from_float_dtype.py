import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('float_dtype', [np.float16, np.float32, np.float64])
def test_astype_from_float_dtype(float_dtype, dtype):
    ser = pd.Series([0.1], dtype=float_dtype)
    result = ser.astype(dtype)
    expected = pd.Series(['0.1'], dtype=dtype)
    tm.assert_series_equal(result, expected)