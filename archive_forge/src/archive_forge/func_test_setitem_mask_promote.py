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
def test_setitem_mask_promote(self):
    ser = Series([0, 'foo', 'bar', 0])
    mask = Series([False, True, True, False])
    ser2 = ser[mask]
    ser[mask] = ser2
    expected = Series([0, 'foo', 'bar', 0])
    tm.assert_series_equal(ser, expected)