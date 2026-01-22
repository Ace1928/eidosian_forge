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
@pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
@pytest.mark.parametrize('utc, args, expected', [pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-08:00'], DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 10:00:00+00:00'], dtype='datetime64[ns, UTC]'), id='all tz-aware, with utc'), pytest.param(False, ['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00']), id='all tz-aware, without utc'), pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[ns, UTC]'), id='all tz-aware, mixed offsets, with utc'), pytest.param(True, ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[ns, UTC]'), id='tz-aware string, naive pydatetime, with utc')])
@pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
def test_to_datetime_mixed_datetime_and_string_with_format(self, fmt, utc, args, expected, constructor):
    ts1 = constructor(args[0])
    ts2 = args[1]
    result = to_datetime([ts1, ts2], format=fmt, utc=utc)
    tm.assert_index_equal(result, expected)