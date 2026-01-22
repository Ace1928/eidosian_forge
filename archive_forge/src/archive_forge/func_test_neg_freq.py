from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_neg_freq(self):
    tdi = pd.timedelta_range('2 Days', periods=4, freq='h')
    arr = tdi._data
    expected = -tdi._data
    result = -arr
    tm.assert_timedelta_array_equal(result, expected)
    result2 = np.negative(arr)
    tm.assert_timedelta_array_equal(result2, expected)