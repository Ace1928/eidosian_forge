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
def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self):
    ser = Series([None] * 10)
    mask = [False] * 3 + [True] * 5 + [False] * 2
    ser[mask] = range(5)
    result = ser
    expected = Series([None] * 3 + list(range(5)) + [None] * 2, dtype=object)
    tm.assert_series_equal(result, expected)