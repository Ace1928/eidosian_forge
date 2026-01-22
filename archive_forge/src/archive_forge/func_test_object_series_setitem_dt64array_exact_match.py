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
def test_object_series_setitem_dt64array_exact_match(self):
    ser = Series({'X': np.nan}, dtype=object)
    indexer = [True]
    value = np.array([4], dtype='M8[ns]')
    ser.iloc[indexer] = value
    expected = Series([value[0]], index=['X'], dtype=object)
    assert all((isinstance(x, np.datetime64) for x in expected.values))
    tm.assert_series_equal(ser, expected)