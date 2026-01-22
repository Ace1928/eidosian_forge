import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_cov_pairwise_diff_length():
    df1 = DataFrame([[1, 5], [3, 2], [3, 9]], columns=Index(['A', 'B'], name='foo'))
    df1a = DataFrame([[1, 5], [3, 9]], index=[0, 2], columns=Index(['A', 'B'], name='foo'))
    df2 = DataFrame([[5, 6], [None, None], [2, 1]], columns=Index(['X', 'Y'], name='foo'))
    df2a = DataFrame([[5, 6], [2, 1]], index=[0, 2], columns=Index(['X', 'Y'], name='foo'))
    result1 = df1.expanding().cov(df2, pairwise=True).loc[2]
    result2 = df1.expanding().cov(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().cov(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().cov(df2a, pairwise=True).loc[2]
    expected = DataFrame([[-3.0, -6.0], [-5.0, -10.0]], columns=Index(['A', 'B'], name='foo'), index=Index(['X', 'Y'], name='foo'))
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)