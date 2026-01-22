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
@pytest.mark.parametrize('errors, expected', [('coerce', Index([NaT, NaT])), ('ignore', Index(['200622-12-31', '111111-24-11'], dtype=object))])
def test_to_datetime_malformed_no_raise(self, errors, expected):
    ts_strings = ['200622-12-31', '111111-24-11']
    with tm.assert_produces_warning(UserWarning, match='Could not infer format', raise_on_extra_warnings=False):
        result = to_datetime(ts_strings, errors=errors)
    tm.assert_index_equal(result, expected)