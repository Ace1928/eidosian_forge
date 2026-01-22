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
@pytest.mark.parametrize('exp, arr, warning', [[['NaT', '2015-06-19 05:33:20', '2015-05-27 22:33:20'], ['foo', 1.434692e+18, 1.432766e+18], UserWarning], [['2015-06-19 05:33:20', '2015-05-27 22:33:20', 'NaT', 'NaT'], [1.434692e+18, 1.432766e+18, 'foo', 'NaT'], None]])
def test_unit_with_numeric_coerce(self, cache, exp, arr, warning):
    expected = DatetimeIndex(exp, dtype='M8[ns]')
    with tm.assert_produces_warning(warning, match='Could not infer format'):
        result = to_datetime(arr, errors='coerce', cache=cache)
    tm.assert_index_equal(result, expected)