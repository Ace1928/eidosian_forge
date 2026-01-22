import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_asarray_tz_aware(self):
    tz = 'US/Central'
    idx = date_range('2000', periods=2, tz=tz)
    expected = np.array(['2000-01-01T06', '2000-01-02T06'], dtype='M8[ns]')
    result = np.asarray(idx, dtype='datetime64[ns]')
    tm.assert_numpy_array_equal(result, expected)
    result = np.asarray(idx, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
    expected = np.array([Timestamp('2000-01-01', tz=tz), Timestamp('2000-01-02', tz=tz)])
    result = np.asarray(idx, dtype=object)
    tm.assert_numpy_array_equal(result, expected)