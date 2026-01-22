from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_properties_daily(self):
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        b_date = Period(freq='B', year=2007, month=1, day=1)
    assert b_date.year == 2007
    assert b_date.quarter == 1
    assert b_date.month == 1
    assert b_date.day == 1
    assert b_date.weekday == 0
    assert b_date.dayofyear == 1
    assert b_date.days_in_month == 31
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert Period(freq='B', year=2012, month=2, day=1).days_in_month == 29
    d_date = Period(freq='D', year=2007, month=1, day=1)
    assert d_date.year == 2007
    assert d_date.quarter == 1
    assert d_date.month == 1
    assert d_date.day == 1
    assert d_date.weekday == 0
    assert d_date.dayofyear == 1
    assert d_date.days_in_month == 31
    assert Period(freq='D', year=2012, month=2, day=1).days_in_month == 29