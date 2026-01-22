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
def test_dti_constructor_small_int(self, any_int_numpy_dtype):
    exp = DatetimeIndex(['1970-01-01 00:00:00.00000000', '1970-01-01 00:00:00.00000001', '1970-01-01 00:00:00.00000002'])
    arr = np.array([0, 10, 20], dtype=any_int_numpy_dtype)
    tm.assert_index_equal(DatetimeIndex(arr), exp)