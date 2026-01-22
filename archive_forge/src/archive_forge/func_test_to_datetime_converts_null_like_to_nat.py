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
@pytest.mark.parametrize('cache', [True, False])
@pytest.mark.parametrize('input', [Series([NaT] * 20 + [None] * 20, dtype='object'), Series([NaT] * 60 + [None] * 60, dtype='object'), Series([None] * 20), Series([None] * 60), Series([''] * 20), Series([''] * 60), Series([pd.NA] * 20), Series([pd.NA] * 60), Series([np.nan] * 20), Series([np.nan] * 60)])
def test_to_datetime_converts_null_like_to_nat(self, cache, input):
    expected = Series([NaT] * len(input), dtype='M8[ns]')
    result = to_datetime(input, cache=cache)
    tm.assert_series_equal(result, expected)