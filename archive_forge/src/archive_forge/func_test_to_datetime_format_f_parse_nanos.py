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
def test_to_datetime_format_f_parse_nanos():
    timestamp = '15/02/2020 02:03:04.123456789'
    timestamp_format = '%d/%m/%Y %H:%M:%S.%f'
    result = to_datetime(timestamp, format=timestamp_format)
    expected = Timestamp(year=2020, month=2, day=15, hour=2, minute=3, second=4, microsecond=123456, nanosecond=789)
    assert result == expected