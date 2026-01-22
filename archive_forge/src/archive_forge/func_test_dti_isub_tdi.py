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
def test_dti_isub_tdi(self, tz_naive_fixture, unit):
    tz = tz_naive_fixture
    dti = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
    tdi = pd.timedelta_range('0 days', periods=10, unit=unit)
    expected = date_range('2017-01-01', periods=10, tz=tz, freq='-1D', unit=unit)
    expected = expected._with_freq(None)
    result = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
    result -= tdi
    tm.assert_index_equal(result, expected)
    dta = dti._data.copy()
    dta -= tdi
    tm.assert_datetime_array_equal(dta, expected._data)
    out = dti._data.copy()
    np.subtract(out, tdi, out=out)
    tm.assert_datetime_array_equal(out, expected._data)
    msg = 'cannot subtract a datelike from a TimedeltaArray'
    with pytest.raises(TypeError, match=msg):
        tdi -= dti
    result = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
    result -= tdi.values
    tm.assert_index_equal(result, expected)
    with pytest.raises(TypeError, match=msg):
        tdi.values -= dti
    with pytest.raises(TypeError, match=msg):
        tdi._values -= dti