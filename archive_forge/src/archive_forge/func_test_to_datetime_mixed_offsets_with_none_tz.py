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
@pytest.mark.parametrize('fmt, expected', [pytest.param('%Y-%m-%d %H:%M:%S%z', Index([Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'), Timestamp('2000-01-02 02:00:00+0200', tz='UTC+02:00'), NaT]), id='ISO8601, non-UTC'), pytest.param('%Y-%d-%m %H:%M:%S%z', Index([Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'), Timestamp('2000-02-01 02:00:00+0200', tz='UTC+02:00'), NaT]), id='non-ISO8601, non-UTC')])
def test_to_datetime_mixed_offsets_with_none_tz(self, fmt, expected):
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=False)
    tm.assert_index_equal(result, expected)