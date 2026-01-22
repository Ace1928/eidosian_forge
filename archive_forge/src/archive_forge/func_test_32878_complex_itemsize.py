from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_32878_complex_itemsize():
    arr = np.arange(5).astype('c8')
    ser = Series(arr)
    val = np.finfo(np.float64).max
    val = val.astype('c16')
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser[0] = val
    assert ser[0] == val
    expected = Series([val, 1, 2, 3, 4], dtype='c16')
    tm.assert_series_equal(ser, expected)