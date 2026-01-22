import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
def test_negative_dates():
    ts = Timestamp('-2000-01-01')
    msg = " not yet supported on Timestamps which are outside the range of Python's standard library. For now, please call the components you need \\(such as `.year` and `.month`\\) and construct your string from there.$"
    func = '^strftime'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.strftime('%Y')
    msg = " not yet supported on Timestamps which are outside the range of Python's standard library. "
    func = '^date'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.date()
    func = '^isocalendar'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.isocalendar()
    func = '^timetuple'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.timetuple()
    func = '^toordinal'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.toordinal()