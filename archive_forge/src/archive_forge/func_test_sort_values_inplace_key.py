import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_inplace_key(self, sort_by_key):
    frame = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
    sorted_df = frame.copy()
    return_value = sorted_df.sort_values(by='A', inplace=True, key=sort_by_key)
    assert return_value is None
    expected = frame.sort_values(by='A', key=sort_by_key)
    tm.assert_frame_equal(sorted_df, expected)
    sorted_df = frame.copy()
    return_value = sorted_df.sort_values(by=1, axis=1, inplace=True, key=sort_by_key)
    assert return_value is None
    expected = frame.sort_values(by=1, axis=1, key=sort_by_key)
    tm.assert_frame_equal(sorted_df, expected)
    sorted_df = frame.copy()
    return_value = sorted_df.sort_values(by='A', ascending=False, inplace=True, key=sort_by_key)
    assert return_value is None
    expected = frame.sort_values(by='A', ascending=False, key=sort_by_key)
    tm.assert_frame_equal(sorted_df, expected)
    sorted_df = frame.copy()
    sorted_df.sort_values(by=['A', 'B'], ascending=False, inplace=True, key=sort_by_key)
    expected = frame.sort_values(by=['A', 'B'], ascending=False, key=sort_by_key)
    tm.assert_frame_equal(sorted_df, expected)