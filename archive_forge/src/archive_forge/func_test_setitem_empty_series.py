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
def test_setitem_empty_series(self):
    key = Timestamp('2012-01-01')
    series = Series(dtype=object)
    series[key] = 47
    expected = Series(47, [key])
    tm.assert_series_equal(series, expected)