import numpy as np
import pytest
from pandas.core.dtypes.generic import ABCIndex
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
def test_astype_dt64():
    arr = pd.array([1, 2, 3, pd.NA]) * 10 ** 9
    result = arr.astype('datetime64[ns]')
    expected = np.array([1, 2, 3, 'NaT'], dtype='M8[s]').astype('M8[ns]')
    tm.assert_numpy_array_equal(result, expected)