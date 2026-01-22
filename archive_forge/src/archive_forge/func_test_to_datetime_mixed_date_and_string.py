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
@pytest.mark.parametrize('format', ['%Y-%m-%d', '%Y-%d-%m'], ids=['ISO8601', 'non-ISO8601'])
def test_to_datetime_mixed_date_and_string(self, format):
    d1 = date(2020, 1, 2)
    res = to_datetime(['2020-01-01', d1], format=format)
    expected = DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='M8[ns]')
    tm.assert_index_equal(res, expected)