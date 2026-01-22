from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_ea_other(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    arr = pd.array([7, pd.NA, 9])
    ser = Series(arr)
    mask = np.ones(df.shape, dtype=bool)
    mask[1, :] = False
    result = df.where(mask, ser, axis=0)
    expected = DataFrame({'A': [1, np.nan, 3], 'B': [4, np.nan, 6]})
    tm.assert_frame_equal(result, expected)
    ser2 = Series(arr[:2], index=['A', 'B'])
    expected = DataFrame({'A': [1, 7, 3], 'B': [4, np.nan, 6]})
    result = df.where(mask, ser2, axis=1)
    tm.assert_frame_equal(result, expected)