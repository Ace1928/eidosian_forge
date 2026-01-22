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
@pytest.mark.parametrize('offset', ['+0', '-1foo', 'UTCbar', ':10', '+01:000:01', ''])
def test_to_datetime_parse_timezone_malformed(self, offset):
    fmt = '%Y-%m-%d %H:%M:%S %z'
    date = '2010-01-01 12:00:00 ' + offset
    msg = '|'.join([f"""^time data ".*" doesn\\'t match format ".*", at position 0. {PARSING_ERR_MSG}$""", f'^unconverted data remains when parsing with format ".*": ".*", at position 0. {PARSING_ERR_MSG}$'])
    with pytest.raises(ValueError, match=msg):
        to_datetime([date], format=fmt)