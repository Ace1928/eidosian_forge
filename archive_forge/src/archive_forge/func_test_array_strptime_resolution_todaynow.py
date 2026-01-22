from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.strptime import array_strptime
from pandas import (
import pandas._testing as tm
def test_array_strptime_resolution_todaynow(self):
    vals = np.array(['today', np.datetime64('2017-01-01', 'us')], dtype=object)
    now = Timestamp('now').asm8
    res, _ = array_strptime(vals, fmt='%Y-%m-%d', utc=False, creso=creso_infer)
    res2, _ = array_strptime(vals[::-1], fmt='%Y-%m-%d', utc=False, creso=creso_infer)
    tolerance = np.timedelta64(1, 's')
    assert res.dtype == 'M8[us]'
    assert abs(res[0] - now) < tolerance
    assert res[1] == vals[1]
    assert res2.dtype == 'M8[us]'
    assert abs(res2[1] - now) < tolerance * 2
    assert res2[0] == vals[1]