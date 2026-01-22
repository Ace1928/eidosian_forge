from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.xfail(not IS64, reason='Floating point error')
def test_to_timedelta_float(self):
    arr = np.arange(0, 1, 1e-06)[-10:]
    result = to_timedelta(arr, unit='s')
    expected_asi8 = np.arange(999990000, 10 ** 9, 1000, dtype='int64')
    tm.assert_numpy_array_equal(result.asi8, expected_asi8)