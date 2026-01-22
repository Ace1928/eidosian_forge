from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_constructor_timestamp_near_dst(self):
    ts = [Timestamp('2016-10-30 03:00:00+0300', tz='Europe/Helsinki'), Timestamp('2016-10-30 03:00:00+0200', tz='Europe/Helsinki')]
    result = DatetimeIndex(ts)
    expected = DatetimeIndex([ts[0].to_pydatetime(), ts[1].to_pydatetime()])
    tm.assert_index_equal(result, expected)