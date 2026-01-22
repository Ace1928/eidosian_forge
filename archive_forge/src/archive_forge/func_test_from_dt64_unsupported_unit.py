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
def test_from_dt64_unsupported_unit(self):
    val = np.datetime64(1, 'D')
    result = DatetimeIndex([val], tz='US/Pacific')
    expected = DatetimeIndex([val.astype('M8[s]')], tz='US/Pacific')
    tm.assert_index_equal(result, expected)