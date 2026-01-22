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
def test_setitem_keep_precision(self, any_numeric_ea_dtype):
    ser = Series([1, 2], dtype=any_numeric_ea_dtype)
    ser[2] = 10
    expected = Series([1, 2, 10], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(ser, expected)