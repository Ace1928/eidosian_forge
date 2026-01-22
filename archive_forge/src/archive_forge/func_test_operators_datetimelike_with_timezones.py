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
def test_operators_datetimelike_with_timezones(self):
    tz = 'US/Eastern'
    dt1 = Series(date_range('2000-01-01 09:00:00', periods=5, tz=tz), name='foo')
    dt2 = dt1.copy()
    dt2.iloc[2] = np.nan
    td1 = Series(pd.timedelta_range('1 days 1 min', periods=5, freq='h'))
    td2 = td1.copy()
    td2.iloc[1] = np.nan
    assert td2._values.freq is None
    result = dt1 + td1[0]
    exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = dt2 + td2[0]
    exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = td1[0] + dt1
    exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = td2[0] + dt2
    exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = dt1 - td1[0]
    exp = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    msg = '(bad|unsupported) operand type for unary'
    with pytest.raises(TypeError, match=msg):
        td1[0] - dt1
    result = dt2 - td2[0]
    exp = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    with pytest.raises(TypeError, match=msg):
        td2[0] - dt2
    result = dt1 + td1
    exp = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = dt2 + td2
    exp = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = dt1 - td1
    exp = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    result = dt2 - td2
    exp = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
    tm.assert_series_equal(result, exp)
    msg = 'cannot (add|subtract)'
    with pytest.raises(TypeError, match=msg):
        td1 - dt1
    with pytest.raises(TypeError, match=msg):
        td2 - dt2