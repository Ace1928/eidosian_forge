import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', indexes_can_append, ids=lambda x: type(x).__name__)
def test_append_same_columns_type(self, index):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=index)
    ser_index = index[:2]
    ser = Series([7, 8], index=ser_index, name=2)
    result = df._append(ser)
    expected = DataFrame([[1, 2, 3.0], [4, 5, 6], [7, 8, np.nan]], index=[0, 1, 2], columns=index)
    assert expected.dtypes.iloc[0].kind == 'i'
    assert expected.dtypes.iloc[1].kind == 'i'
    tm.assert_frame_equal(result, expected)
    ser_index = index
    index = index[:2]
    df = DataFrame([[1, 2], [4, 5]], columns=index)
    ser = Series([7, 8, 9], index=ser_index, name=2)
    result = df._append(ser)
    expected = DataFrame([[1, 2, np.nan], [4, 5, np.nan], [7, 8, 9]], index=[0, 1, 2], columns=ser_index)
    tm.assert_frame_equal(result, expected)