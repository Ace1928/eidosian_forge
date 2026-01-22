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
@pytest.mark.parametrize('date', [['2020-10-26 00:00:00+06:00', '2020-10-26 00:00:00+01:00'], ['2020-10-26 00:00:00+06:00', Timestamp('2018-01-01', tz='US/Pacific')], ['2020-10-26 00:00:00+06:00', datetime(2020, 1, 1, 18, tzinfo=pytz.timezone('Australia/Melbourne'))]])
def test_to_datetime_mixed_offsets_with_utc_false_deprecated(self, date):
    msg = 'parsing datetimes with mixed time zones will raise an error'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        to_datetime(date, utc=False)