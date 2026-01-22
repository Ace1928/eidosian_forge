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
@td.skip_if_not_us_locale
@pytest.mark.parametrize('errors', ['raise', 'coerce', 'ignore'])
def test_to_datetime_with_apply_with_empty_str(self, cache, errors):
    td = Series(['May 04', 'Jun 02', ''], index=[1, 2, 3])
    expected = to_datetime(td, format='%b %y', errors=errors, cache=cache)
    result = td.apply(lambda x: to_datetime(x, format='%b %y', errors='coerce', cache=cache))
    tm.assert_series_equal(result, expected)