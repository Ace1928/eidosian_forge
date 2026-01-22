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
def test_sub_timedelta64_mismatched_reso(self, ts_tz):
    ts = ts_tz
    res = ts + np.timedelta64(1, 'ns')
    exp = ts.as_unit('ns') + np.timedelta64(1, 'ns')
    assert exp == res
    assert exp._creso == NpyDatetimeUnit.NPY_FR_ns.value