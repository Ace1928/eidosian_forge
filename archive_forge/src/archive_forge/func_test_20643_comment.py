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
def test_20643_comment():
    orig = Series([0, 1, 2], index=['a', 'b', 'c'])
    expected = Series([np.nan, 1, 2], index=['a', 'b', 'c'])
    ser = orig.copy()
    ser.iat[0] = None
    tm.assert_series_equal(ser, expected)
    ser = orig.copy()
    ser.iloc[0] = None
    tm.assert_series_equal(ser, expected)