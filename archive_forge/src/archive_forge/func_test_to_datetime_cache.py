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
@pytest.mark.parametrize('utc', [True, None])
@pytest.mark.parametrize('format', ['%Y%m%d %H:%M:%S', None])
@pytest.mark.parametrize('constructor', [list, tuple, np.array, Index, deque])
def test_to_datetime_cache(self, utc, format, constructor):
    date = '20130101 00:00:00'
    test_dates = [date] * 10 ** 5
    data = constructor(test_dates)
    result = to_datetime(data, utc=utc, format=format, cache=True)
    expected = to_datetime(data, utc=utc, format=format, cache=False)
    tm.assert_index_equal(result, expected)