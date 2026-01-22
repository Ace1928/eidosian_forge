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
@pytest.mark.skip_ubsan
def test_to_datetime_dt64d_out_of_bounds(self, cache):
    dt64 = np.datetime64(np.iinfo(np.int64).max, 'D')
    msg = 'Out of bounds second timestamp: 25252734927768524-07-27'
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp(dt64)
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        to_datetime(dt64, errors='raise', cache=cache)
    assert to_datetime(dt64, errors='coerce', cache=cache) is NaT