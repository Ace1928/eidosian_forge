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
@pytest.mark.parametrize('years', [-1, 0, 1])
@pytest.mark.parametrize('months', [-2, 0, 2])
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_shift_months(years, months, unit):
    dti = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-02-29'), Timestamp('2000-12-31')]).as_unit(unit)
    shifted = shift_months(dti.asi8, years * 12 + months, reso=dti._data._creso)
    shifted_dt64 = shifted.view(f'M8[{dti.unit}]')
    actual = DatetimeIndex(shifted_dt64)
    raw = [x + pd.offsets.DateOffset(years=years, months=months) for x in dti]
    expected = DatetimeIndex(raw).as_unit(dti.unit)
    tm.assert_index_equal(actual, expected)