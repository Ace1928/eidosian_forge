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
def test_out_of_ns_bounds(self):
    result = Timestamp(-52700112000, unit='s')
    assert result == Timestamp('0300-01-01')
    assert result.to_numpy() == np.datetime64('0300-01-01T00:00:00', 's')