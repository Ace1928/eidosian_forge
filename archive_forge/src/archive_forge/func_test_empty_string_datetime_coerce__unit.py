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
def test_empty_string_datetime_coerce__unit():
    result = to_datetime([1, ''], unit='s', errors='coerce')
    expected = DatetimeIndex(['1970-01-01 00:00:01', 'NaT'], dtype='datetime64[ns]')
    tm.assert_index_equal(expected, result)
    result = to_datetime([1, ''], unit='s', errors='raise')
    tm.assert_index_equal(expected, result)