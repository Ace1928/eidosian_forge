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
@pytest.mark.parametrize('args', [pytest.param(['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-07:00'], id='all tz-aware, mixed timezones, without utc')])
@pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
def test_to_datetime_mixed_datetime_and_string_with_format_raises(self, fmt, args, constructor):
    ts1 = constructor(args[0])
    ts2 = constructor(args[1])
    with pytest.raises(ValueError, match='cannot be converted to datetime64 unless utc=True'):
        to_datetime([ts1, ts2], format=fmt, utc=False)