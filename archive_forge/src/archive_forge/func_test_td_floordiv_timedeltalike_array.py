from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_floordiv_timedeltalike_array(self):
    td = Timedelta(hours=3, minutes=4)
    scalar = Timedelta(hours=3, minutes=3)
    assert td // np.array(scalar.to_timedelta64()) == 1
    res = 3 * td // np.array([scalar.to_timedelta64()])
    expected = np.array([3], dtype=np.int64)
    tm.assert_numpy_array_equal(res, expected)
    res = 10 * td // np.array([scalar.to_timedelta64(), np.timedelta64('NaT')])
    expected = np.array([10, np.nan])
    tm.assert_numpy_array_equal(res, expected)