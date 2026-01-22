import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
@pytest.mark.parametrize('dt_string, tz, dt_string_repr', [('2013-01-01 05:45+0545', timezone(timedelta(minutes=345)), "Timestamp('2013-01-01 05:45:00+0545', tz='UTC+05:45')"), ('2013-01-01 05:30+0530', timezone(timedelta(minutes=330)), "Timestamp('2013-01-01 05:30:00+0530', tz='UTC+05:30')")])
def test_parsers_timezone_minute_offsets_roundtrip(self, cache, dt_string, tz, dt_string_repr):
    base = to_datetime('2013-01-01 00:00:00', cache=cache)
    base = base.tz_localize('UTC').tz_convert(tz)
    dt_time = to_datetime(dt_string, cache=cache)
    assert base == dt_time
    assert dt_string_repr == repr(dt_time)