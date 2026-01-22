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
@pytest.mark.parametrize('input, format', [('2012-01-01', '%Y-%m'), ('2012-01-01 10', '%Y-%m-%d'), ('2012-01-01 10:00', '%Y-%m-%d %H'), ('2012-01-01 10:00:00', '%Y-%m-%d %H:%M'), (0, '%Y-%m-%d')])
def test_to_datetime_iso8601_exact_fails(self, input, format):
    msg = '|'.join([f'^unconverted data remains when parsing with format ".*": ".*", at position 0. {PARSING_ERR_MSG}$', f"""^time data ".*" doesn't match format ".*", at position 0. {PARSING_ERR_MSG}$"""])
    with pytest.raises(ValueError, match=msg):
        to_datetime(input, format=format)