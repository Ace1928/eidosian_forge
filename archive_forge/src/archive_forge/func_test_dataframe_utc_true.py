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
def test_dataframe_utc_true(self):
    df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
    result = to_datetime(df, utc=True)
    expected = Series(np.array(['2015-02-04', '2016-03-05'], dtype='datetime64[ns]')).dt.tz_localize('UTC')
    tm.assert_series_equal(result, expected)