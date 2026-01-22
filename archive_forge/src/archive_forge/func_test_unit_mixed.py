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
@pytest.mark.parametrize('arr', [[Timestamp('20130101'), 1.434692e+18, 1.432766e+18], [1.434692e+18, 1.432766e+18, Timestamp('20130101')]])
def test_unit_mixed(self, cache, arr):
    expected = Index([Timestamp(x) for x in arr], dtype='M8[ns]')
    result = to_datetime(arr, errors='coerce', cache=cache)
    tm.assert_index_equal(result, expected)
    result = to_datetime(arr, errors='raise', cache=cache)
    tm.assert_index_equal(result, expected)
    result = DatetimeIndex(arr)
    tm.assert_index_equal(result, expected)