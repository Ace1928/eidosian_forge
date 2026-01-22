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
def test_to_datetime_different_offsets(self, cache):
    ts_string_1 = 'March 1, 2018 12:00:00+0400'
    ts_string_2 = 'March 1, 2018 12:00:00+0500'
    arr = [ts_string_1] * 5 + [ts_string_2] * 5
    expected = Index([parse(x) for x in arr])
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_datetime(arr, cache=cache)
    tm.assert_index_equal(result, expected)