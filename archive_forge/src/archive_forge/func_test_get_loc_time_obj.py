from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_time_obj(self):
    idx = date_range('2000-01-01', periods=24, freq='h')
    result = idx.get_loc(time(12))
    expected = np.array([12])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)
    result = idx.get_loc(time(12, 30))
    expected = np.array([])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)