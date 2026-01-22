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
def test_setitem_categorical_assigning_ops():
    orig = Series(Categorical(['b', 'b'], categories=['a', 'b']))
    ser = orig.copy()
    ser[:] = 'a'
    exp = Series(Categorical(['a', 'a'], categories=['a', 'b']))
    tm.assert_series_equal(ser, exp)
    ser = orig.copy()
    ser[1] = 'a'
    exp = Series(Categorical(['b', 'a'], categories=['a', 'b']))
    tm.assert_series_equal(ser, exp)
    ser = orig.copy()
    ser[ser.index > 0] = 'a'
    exp = Series(Categorical(['b', 'a'], categories=['a', 'b']))
    tm.assert_series_equal(ser, exp)
    ser = orig.copy()
    ser[[False, True]] = 'a'
    exp = Series(Categorical(['b', 'a'], categories=['a', 'b']))
    tm.assert_series_equal(ser, exp)
    ser = orig.copy()
    ser.index = ['x', 'y']
    ser['y'] = 'a'
    exp = Series(Categorical(['b', 'a'], categories=['a', 'b']), index=['x', 'y'])
    tm.assert_series_equal(ser, exp)