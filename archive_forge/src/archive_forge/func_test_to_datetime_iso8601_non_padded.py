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
@pytest.mark.parametrize('input, format', [('2020-1', '%Y-%m'), ('2020-1-1', '%Y-%m-%d'), ('2020-1-1 0', '%Y-%m-%d %H'), ('2020-1-1T0', '%Y-%m-%dT%H'), ('2020-1-1 0:0', '%Y-%m-%d %H:%M'), ('2020-1-1T0:0', '%Y-%m-%dT%H:%M'), ('2020-1-1 0:0:0', '%Y-%m-%d %H:%M:%S'), ('2020-1-1T0:0:0', '%Y-%m-%dT%H:%M:%S'), ('2020-1-1T0:0:0.000', '%Y-%m-%dT%H:%M:%S.%f'), ('2020-1-1T0:0:0.000000', '%Y-%m-%dT%H:%M:%S.%f'), ('2020-1-1T0:0:0.000000000', '%Y-%m-%dT%H:%M:%S.%f')])
def test_to_datetime_iso8601_non_padded(self, input, format):
    expected = Timestamp(2020, 1, 1)
    result = to_datetime(input, format=format)
    assert result == expected