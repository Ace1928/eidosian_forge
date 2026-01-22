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
def test_properties_hourly(self):
    h_date1 = Period(freq='h', year=2007, month=1, day=1, hour=0)
    h_date2 = Period(freq='2h', year=2007, month=1, day=1, hour=0)
    for h_date in [h_date1, h_date2]:
        assert h_date.year == 2007
        assert h_date.quarter == 1
        assert h_date.month == 1
        assert h_date.day == 1
        assert h_date.weekday == 0
        assert h_date.dayofyear == 1
        assert h_date.hour == 0
        assert h_date.days_in_month == 31
        assert Period(freq='h', year=2012, month=2, day=1, hour=0).days_in_month == 29