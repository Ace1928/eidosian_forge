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
def test_setitem_mask_smallint_upcast(self):
    orig = Series([1, 2, 3], dtype='int8')
    alt = np.array([999, 1000, 1001], dtype=np.int64)
    mask = np.array([True, False, True])
    ser = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        ser[mask] = Series(alt)
    expected = Series([999, 2, 1001])
    tm.assert_series_equal(ser, expected)
    ser2 = orig.copy()
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        ser2.mask(mask, alt, inplace=True)
    tm.assert_series_equal(ser2, expected)
    ser3 = orig.copy()
    res = ser3.where(~mask, Series(alt))
    tm.assert_series_equal(res, expected)