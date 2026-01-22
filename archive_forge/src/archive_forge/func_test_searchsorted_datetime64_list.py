import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
def test_searchsorted_datetime64_list(self):
    ser = Series(date_range('20120101', periods=10, freq='2D'))
    vals = [Timestamp('20120102'), Timestamp('20120104')]
    res = ser.searchsorted(vals)
    exp = np.array([1, 2], dtype=np.intp)
    tm.assert_numpy_array_equal(res, exp)