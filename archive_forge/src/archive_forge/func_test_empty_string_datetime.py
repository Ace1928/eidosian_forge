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
@pytest.mark.parametrize('errors', ['raise', 'coerce', 'ignore'])
@pytest.mark.parametrize('args, format', [(['03/24/2016', '03/25/2016', ''], '%m/%d/%Y'), (['2016-03-24', '2016-03-25', ''], '%Y-%m-%d')], ids=['non-ISO8601', 'ISO8601'])
def test_empty_string_datetime(errors, args, format):
    td = Series(args)
    result = to_datetime(td, format=format, errors=errors)
    expected = Series(['2016-03-24', '2016-03-25', NaT], dtype='datetime64[ns]')
    tm.assert_series_equal(expected, result)