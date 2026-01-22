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
def test_setitem_callable_key(self):
    ser = Series([1, 2, 3, 4], index=list('ABCD'))
    ser[lambda x: 'A'] = -1
    expected = Series([-1, 2, 3, 4], index=list('ABCD'))
    tm.assert_series_equal(ser, expected)