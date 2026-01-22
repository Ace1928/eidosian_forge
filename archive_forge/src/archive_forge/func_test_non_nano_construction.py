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
def test_non_nano_construction(self, dt64, ts, reso):
    assert ts._value == dt64.view('i8')
    if reso == 's':
        assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value
    elif reso == 'ms':
        assert ts._creso == NpyDatetimeUnit.NPY_FR_ms.value
    elif reso == 'us':
        assert ts._creso == NpyDatetimeUnit.NPY_FR_us.value