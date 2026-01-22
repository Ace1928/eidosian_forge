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
def test_default_to_stdlib_utc(self):
    assert Timestamp.utcnow().tz is timezone.utc
    assert Timestamp.now('UTC').tz is timezone.utc
    assert Timestamp('2016-01-01', tz='UTC').tz is timezone.utc