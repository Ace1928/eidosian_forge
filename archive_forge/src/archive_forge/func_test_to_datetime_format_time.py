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
@pytest.mark.parametrize('value, format, dt', [['01/10/2010 15:20', '%m/%d/%Y %H:%M', Timestamp('2010-01-10 15:20')], ['01/10/2010 05:43', '%m/%d/%Y %I:%M', Timestamp('2010-01-10 05:43')], ['01/10/2010 13:56:01', '%m/%d/%Y %H:%M:%S', Timestamp('2010-01-10 13:56:01')], pytest.param('01/10/2010 08:14 PM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 20:14'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False)), pytest.param('01/10/2010 07:40 AM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 07:40'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False)), pytest.param('01/10/2010 09:12:56 AM', '%m/%d/%Y %I:%M:%S %p', Timestamp('2010-01-10 09:12:56'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False))])
def test_to_datetime_format_time(self, cache, value, format, dt):
    assert to_datetime(value, format=format, cache=cache) == dt