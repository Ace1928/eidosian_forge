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
@pytest.mark.parametrize('fmt, expected', [pytest.param('%Y-%m-%d %H:%M:%S%z', DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-01-02 00:00:00+00:00', 'NaT'], dtype='datetime64[ns, UTC]'), id='ISO8601, UTC'), pytest.param('%Y-%d-%m %H:%M:%S%z', DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-02-01 00:00:00+00:00', 'NaT'], dtype='datetime64[ns, UTC]'), id='non-ISO8601, UTC')])
def test_to_datetime_mixed_offsets_with_none(self, fmt, expected):
    result = to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=True)
    tm.assert_index_equal(result, expected)