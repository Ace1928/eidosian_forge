from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('freq', ['h', 'D', 'W', '2ME', 'MS', 'QE', 'B', None])
@pytest.mark.parametrize('dtype', [None, 'uint8'])
def test_dt64arr_addsub_intlike(self, request, dtype, index_or_series_or_array, freq, tz_naive_fixture):
    tz = tz_naive_fixture
    if freq is None:
        dti = DatetimeIndex(['NaT', '2017-04-05 06:07:08'], tz=tz)
    else:
        dti = date_range('2016-01-01', periods=2, freq=freq, tz=tz)
    obj = index_or_series_or_array(dti)
    other = np.array([4, -1])
    if dtype is not None:
        other = other.astype(dtype)
    msg = '|'.join(['Addition/subtraction of integers', 'cannot subtract DatetimeArray from', 'can only perform ops with numeric values', 'unsupported operand type.*Categorical', "unsupported operand type\\(s\\) for -: 'int' and 'Timestamp'"])
    assert_invalid_addsub_type(obj, 1, msg)
    assert_invalid_addsub_type(obj, np.int64(2), msg)
    assert_invalid_addsub_type(obj, np.array(3, dtype=np.int64), msg)
    assert_invalid_addsub_type(obj, other, msg)
    assert_invalid_addsub_type(obj, np.array(other), msg)
    assert_invalid_addsub_type(obj, pd.array(other), msg)
    assert_invalid_addsub_type(obj, pd.Categorical(other), msg)
    assert_invalid_addsub_type(obj, pd.Index(other), msg)
    assert_invalid_addsub_type(obj, Series(other), msg)