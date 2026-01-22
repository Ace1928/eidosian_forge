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
def test_to_datetime_with_space_in_series(self, cache):
    ser = Series(['10/18/2006', '10/18/2008', ' '])
    msg = f"""^time data " " doesn\\'t match format "%m/%d/%Y", at position 2. {PARSING_ERR_MSG}$"""
    with pytest.raises(ValueError, match=msg):
        to_datetime(ser, errors='raise', cache=cache)
    result_coerce = to_datetime(ser, errors='coerce', cache=cache)
    expected_coerce = Series([datetime(2006, 10, 18), datetime(2008, 10, 18), NaT])
    tm.assert_series_equal(result_coerce, expected_coerce)
    result_ignore = to_datetime(ser, errors='ignore', cache=cache)
    tm.assert_series_equal(result_ignore, ser)