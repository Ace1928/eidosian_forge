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
@pytest.mark.xfail(reason='Dispatches to np.datetime64 which is wrong')
def test_cmp_cross_reso_reversed_dt64(self):
    dt64 = np.datetime64(106752, 'D')
    ts = Timestamp._from_dt64(dt64)
    other = Timestamp(dt64 - 1)
    assert other.asm8 < ts