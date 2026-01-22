from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@td.skip_if_windows
def test_month_range_union_tz_dateutil(self, sort):
    from pandas._libs.tslibs.timezones import dateutil_gettz
    tz = dateutil_gettz('US/Eastern')
    early_start = datetime(2011, 1, 1)
    early_end = datetime(2011, 3, 1)
    late_start = datetime(2011, 3, 1)
    late_end = datetime(2011, 5, 1)
    early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
    late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())
    early_dr.union(late_dr, sort=sort)