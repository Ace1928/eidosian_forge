from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_left_different_named_levels():
    left = Series([2], index=pd.MultiIndex.from_tuples([(1, 4, 3)], names=['a', 'd', 'c']))
    right = Series([1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=['a', 'b', 'c']))
    result_left, result_right = left.align(right)
    expected_left = Series([2], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=['a', 'd', 'c', 'b']))
    expected_right = Series([1], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=['a', 'd', 'c', 'b']))
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)