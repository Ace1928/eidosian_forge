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
def test_15413():
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser[ser == 2] += 0.5
    expected = Series([1, 2.5, 3])
    tm.assert_series_equal(ser, expected)
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser[1] += 0.5
    tm.assert_series_equal(ser, expected)
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.loc[1] += 0.5
    tm.assert_series_equal(ser, expected)
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.iloc[1] += 0.5
    tm.assert_series_equal(ser, expected)
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.iat[1] += 0.5
    tm.assert_series_equal(ser, expected)
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        ser.at[1] += 0.5
    tm.assert_series_equal(ser, expected)