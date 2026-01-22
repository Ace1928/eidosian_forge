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
@pytest.mark.parametrize('data, format, expected', [([pd.NA], '%Y%m%d%H%M%S', DatetimeIndex(['NaT'])), ([pd.NA], None, DatetimeIndex(['NaT'])), ([pd.NA, '20210202202020'], '%Y%m%d%H%M%S', DatetimeIndex(['NaT', '2021-02-02 20:20:20'])), (['201010', pd.NA], '%y%m%d', DatetimeIndex(['2020-10-10', 'NaT'])), (['201010', pd.NA], '%d%m%y', DatetimeIndex(['2010-10-20', 'NaT'])), ([None, np.nan, pd.NA], None, DatetimeIndex(['NaT', 'NaT', 'NaT'])), ([None, np.nan, pd.NA], '%Y%m%d', DatetimeIndex(['NaT', 'NaT', 'NaT']))])
def test_to_datetime_with_NA(self, data, format, expected):
    result = to_datetime(data, format=format)
    tm.assert_index_equal(result, expected)