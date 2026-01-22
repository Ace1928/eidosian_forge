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
@pytest.mark.parametrize('func', [list, np.array, Series])
def test_setitem_boolean_python_list(self, func):
    ser = Series([None, 'b', None])
    mask = func([True, False, True])
    ser[mask] = ['a', 'c']
    expected = Series(['a', 'b', 'c'])
    tm.assert_series_equal(ser, expected)