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
def test_datetime_bool_arrays_mixed(self, cache):
    msg = f'{type(cache)} is not convertible to datetime'
    with pytest.raises(TypeError, match=msg):
        to_datetime([False, datetime.today()], cache=cache)
    with pytest.raises(ValueError, match=f"""^time data "True" doesn\\'t match format "%Y%m%d", at position 1. {PARSING_ERR_MSG}$"""):
        to_datetime(['20130101', True], cache=cache)
    tm.assert_index_equal(to_datetime([0, False, NaT, 0.0], errors='coerce', cache=cache), DatetimeIndex([to_datetime(0, cache=cache), NaT, NaT, to_datetime(0, cache=cache)]))