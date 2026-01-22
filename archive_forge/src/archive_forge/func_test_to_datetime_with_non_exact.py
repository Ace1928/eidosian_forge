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
@td.skip_if_not_us_locale
def test_to_datetime_with_non_exact(self, cache):
    ser = Series(['19MAY11', 'foobar19MAY11', '19MAY11:00:00:00', '19MAY11 00:00:00Z'])
    result = to_datetime(ser, format='%d%b%y', exact=False, cache=cache)
    expected = to_datetime(ser.str.extract('(\\d+\\w+\\d+)', expand=False), format='%d%b%y', cache=cache)
    tm.assert_series_equal(result, expected)