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
@pytest.mark.parametrize('datetimelikes,expected_values', (((None, np.nan) + (NaT,) * start_caching_at, (NaT,) * (start_caching_at + 2)), ((None, Timestamp('2012-07-26')) + (NaT,) * start_caching_at, (NaT, Timestamp('2012-07-26')) + (NaT,) * start_caching_at), ((None,) + (NaT,) * start_caching_at + ('2012 July 26', Timestamp('2012-07-26')), (NaT,) * (start_caching_at + 1) + (Timestamp('2012-07-26'), Timestamp('2012-07-26')))))
def test_convert_object_to_datetime_with_cache(self, datetimelikes, expected_values):
    ser = Series(datetimelikes, dtype='object')
    result_series = to_datetime(ser, errors='coerce')
    expected_series = Series(expected_values, dtype='datetime64[ns]')
    tm.assert_series_equal(result_series, expected_series)