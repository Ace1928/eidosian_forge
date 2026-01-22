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
@pytest.mark.parametrize('test_list', [['2011-12-30 00:00:00.000000', '2011-12-30 00:00:00.000000', '2011-12-30 00:00:00.000000'], [np.nan, np.nan, '2011-12-30 00:00:00.000000'], ['', '2011-12-30 00:00:00.000000'], ['NaT', '2011-12-30 00:00:00.000000'], ['2011-12-30 00:00:00.000000', 'random_string'], ['now', '2011-12-30 00:00:00.000000'], ['today', '2011-12-30 00:00:00.000000']])
def test_guess_datetime_format_for_array(self, test_list):
    expected_format = '%Y-%m-%d %H:%M:%S.%f'
    test_array = np.array(test_list, dtype=object)
    assert tools._guess_datetime_format_for_array(test_array) == expected_format