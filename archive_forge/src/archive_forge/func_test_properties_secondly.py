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
def test_properties_secondly(self):
    s_date = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0, second=0)
    assert s_date.year == 2007
    assert s_date.quarter == 1
    assert s_date.month == 1
    assert s_date.day == 1
    assert s_date.weekday == 0
    assert s_date.dayofyear == 1
    assert s_date.hour == 0
    assert s_date.minute == 0
    assert s_date.second == 0
    assert s_date.days_in_month == 31
    assert Period(freq='Min', year=2012, month=2, day=1, hour=0, minute=0, second=0).days_in_month == 29