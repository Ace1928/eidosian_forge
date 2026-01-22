from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
def test_infer_homogeoneous_dt64(self):
    dt = datetime(2023, 10, 27, 18, 3, 5, 678000)
    dt64 = np.datetime64(dt, 'ms')
    arr = np.array([None, dt64, dt64, dt64], dtype=object)
    result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
    assert tz is None
    expected = np.array([np.datetime64('NaT'), dt64, dt64, dt64], dtype='M8[ms]')
    tm.assert_numpy_array_equal(result, expected)