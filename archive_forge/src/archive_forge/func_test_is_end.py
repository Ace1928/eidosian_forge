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
@pytest.mark.parametrize('end', ['is_month_end', 'is_year_end', 'is_quarter_end'])
@pytest.mark.parametrize('tz', [None, 'US/Eastern'])
def test_is_end(self, end, tz):
    ts = Timestamp('2014-12-31 23:59:59', tz=tz)
    assert getattr(ts, end)