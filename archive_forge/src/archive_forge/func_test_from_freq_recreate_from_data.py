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
@pytest.mark.parametrize('freq', ['ME', 'QE', 'YE', 'D', 'B', 'bh', 'min', 's', 'ms', 'us', 'h', 'ns', 'C'])
def test_from_freq_recreate_from_data(self, freq):
    org = date_range(start='2001/02/01 09:00', freq=freq, periods=1)
    idx = DatetimeIndex(org, freq=freq)
    tm.assert_index_equal(idx, org)
    org = date_range(start='2001/02/01 09:00', freq=freq, tz='US/Pacific', periods=1)
    idx = DatetimeIndex(org, freq=freq, tz='US/Pacific')
    tm.assert_index_equal(idx, org)