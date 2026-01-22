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
def test_dayfirst_warnings_valid_input(self):
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    arr = ['31/12/2014', '10/03/2011']
    expected = DatetimeIndex(['2014-12-31', '2011-03-10'], dtype='datetime64[ns]', freq=None)
    res1 = to_datetime(arr, dayfirst=True)
    tm.assert_index_equal(expected, res1)
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res2 = to_datetime(arr, dayfirst=False)
    tm.assert_index_equal(expected, res2)