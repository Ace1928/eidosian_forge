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
@pytest.mark.parametrize('format, expected_ds', [('%Y-%m-%d %H:%M:%S%z', '2020-01-03'), ('%Y-%d-%m %H:%M:%S%z', '2020-03-01'), (None, '2020-01-03')])
@pytest.mark.parametrize('string, attribute', [('now', 'utcnow'), ('today', 'today')])
def test_to_datetime_now_with_format(self, format, expected_ds, string, attribute):
    result = to_datetime(['2020-01-03 00:00:00Z', string], format=format, utc=True)
    expected = DatetimeIndex([expected_ds, getattr(Timestamp, attribute)()], dtype='datetime64[ns, UTC]')
    assert (expected - result).max().total_seconds() < 1