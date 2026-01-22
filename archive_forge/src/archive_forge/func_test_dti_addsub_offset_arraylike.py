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
@pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
def test_dti_addsub_offset_arraylike(self, tz_naive_fixture, names, op, index_or_series):
    other_box = index_or_series
    tz = tz_naive_fixture
    dti = date_range('2017-01-01', periods=2, tz=tz, name=names[0])
    other = other_box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], name=names[1])
    xbox = get_upcast_box(dti, other)
    with tm.assert_produces_warning(PerformanceWarning):
        res = op(dti, other)
    expected = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))], name=names[2], freq='infer')
    expected = tm.box_expected(expected, xbox).astype(object)
    tm.assert_equal(res, expected)