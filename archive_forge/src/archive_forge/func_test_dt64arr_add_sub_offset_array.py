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
@pytest.mark.parametrize('other', [np.array([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)]), np.array([pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()]), np.array([pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)])])
@pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
def test_dt64arr_add_sub_offset_array(self, tz_naive_fixture, box_with_array, op, other):
    tz = tz_naive_fixture
    dti = date_range('2017-01-01', periods=2, tz=tz)
    dtarr = tm.box_expected(dti, box_with_array)
    expected = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
    expected = tm.box_expected(expected, box_with_array).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res = op(dtarr, other)
    tm.assert_equal(res, expected)
    other = tm.box_expected(other, box_with_array)
    if box_with_array is pd.array and op is roperator.radd:
        expected = pd.array(expected, dtype=object)
    with tm.assert_produces_warning(PerformanceWarning):
        res = op(dtarr, other)
    tm.assert_equal(res, expected)