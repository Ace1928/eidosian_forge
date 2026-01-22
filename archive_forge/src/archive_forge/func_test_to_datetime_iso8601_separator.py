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
@pytest.mark.parametrize('input, format', [('2020-01', '%Y/%m'), ('2020-01-01', '%Y/%m/%d'), ('2020-01-01 00', '%Y/%m/%dT%H'), ('2020-01-01T00', '%Y/%m/%d %H'), ('2020-01-01 00:00', '%Y/%m/%dT%H:%M'), ('2020-01-01T00:00', '%Y/%m/%d %H:%M'), ('2020-01-01 00:00:00', '%Y/%m/%dT%H:%M:%S'), ('2020-01-01T00:00:00', '%Y/%m/%d %H:%M:%S')])
def test_to_datetime_iso8601_separator(self, input, format):
    with pytest.raises(ValueError, match=f'''time data \\"{input}\\" doesn\\'t match format \\"{format}\\", at position 0'''):
        to_datetime(input, format=format)