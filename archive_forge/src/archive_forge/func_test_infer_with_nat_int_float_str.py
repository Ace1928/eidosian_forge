from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('item', [float('nan'), NaT.value, float(NaT.value), 'NaT', ''])
def test_infer_with_nat_int_float_str(self, item):
    dt = datetime(2023, 11, 15, 15, 5, 6)
    arr = np.array([dt, item], dtype=object)
    result, tz = tslib.array_to_datetime(arr, creso=creso_infer)
    assert tz is None
    expected = np.array([dt, np.datetime64('NaT')], dtype='M8[us]')
    tm.assert_numpy_array_equal(result, expected)
    result2, tz2 = tslib.array_to_datetime(arr[::-1], creso=creso_infer)
    assert tz2 is None
    tm.assert_numpy_array_equal(result2, expected[::-1])