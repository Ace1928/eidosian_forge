import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_shift_with_multiple_periods():
    df = DataFrame({'a': [1, 2, 3, 3, 2], 'b': [True, True, False, False, True]})
    shifted_df = df.groupby('b')[['a']].shift([0, 1])
    expected_df = DataFrame({'a_0': [1, 2, 3, 3, 2], 'a_1': [np.nan, 1.0, np.nan, 3.0, 2.0]})
    tm.assert_frame_equal(shifted_df, expected_df)
    shifted_series = df.groupby('b')['a'].shift([0, 1])
    tm.assert_frame_equal(shifted_series, expected_df)