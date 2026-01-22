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
def test_dt64_series_arith_overflow(self):
    dt = Timestamp('1700-01-31')
    td = Timedelta('20000 Days')
    dti = date_range('1949-09-30', freq='100YE', periods=4)
    ser = Series(dti)
    msg = 'Overflow in int64 addition'
    with pytest.raises(OverflowError, match=msg):
        ser - dt
    with pytest.raises(OverflowError, match=msg):
        dt - ser
    with pytest.raises(OverflowError, match=msg):
        ser + td
    with pytest.raises(OverflowError, match=msg):
        td + ser
    ser.iloc[-1] = NaT
    expected = Series(['2004-10-03', '2104-10-04', '2204-10-04', 'NaT'], dtype='datetime64[ns]')
    res = ser + td
    tm.assert_series_equal(res, expected)
    res = td + ser
    tm.assert_series_equal(res, expected)
    ser.iloc[1:] = NaT
    expected = Series(['91279 Days', 'NaT', 'NaT', 'NaT'], dtype='timedelta64[ns]')
    res = ser - dt
    tm.assert_series_equal(res, expected)
    res = dt - ser
    tm.assert_series_equal(res, -expected)