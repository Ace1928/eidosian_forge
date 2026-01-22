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
@pytest.mark.parametrize('index', [date_range('2016-01-01', periods=5, tz='US/Pacific'), pd.timedelta_range('1 Day', periods=5)])
def test_shallow_copy_inherits_array_freq(self, index):
    array = index._data
    arr = array[[0, 3, 2, 4, 1]]
    assert arr.freq is None
    result = index._shallow_copy(arr)
    assert result.freq is None