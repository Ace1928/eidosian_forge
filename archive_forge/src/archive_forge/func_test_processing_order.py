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
@pytest.mark.parametrize('arg, origin, expected_str', [[200 * 365, 'unix', '2169-11-13 00:00:00'], [200 * 365, '1870-01-01', '2069-11-13 00:00:00'], [300 * 365, '1870-01-01', '2169-10-20 00:00:00']])
def test_processing_order(self, arg, origin, expected_str):
    result = to_datetime(arg, unit='D', origin=origin)
    expected = Timestamp(expected_str)
    assert result == expected
    result = to_datetime(200 * 365, unit='D', origin='1870-01-01')
    expected = Timestamp('2069-11-13 00:00:00')
    assert result == expected
    result = to_datetime(300 * 365, unit='D', origin='1870-01-01')
    expected = Timestamp('2169-10-20 00:00:00')
    assert result == expected