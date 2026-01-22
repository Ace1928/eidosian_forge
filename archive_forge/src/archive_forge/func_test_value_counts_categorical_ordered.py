import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_value_counts_categorical_ordered(self):
    values = Categorical([1, 2, 3, 1, 1, 3], ordered=True)
    exp_idx = CategoricalIndex([1, 3, 2], categories=[1, 2, 3], ordered=True, name='xxx')
    exp = Series([3, 2, 1], index=exp_idx, name='count')
    ser = Series(values, name='xxx')
    tm.assert_series_equal(ser.value_counts(), exp)
    idx = CategoricalIndex(values, name='xxx')
    tm.assert_series_equal(idx.value_counts(), exp)
    exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name='proportion')
    tm.assert_series_equal(ser.value_counts(normalize=True), exp)
    tm.assert_series_equal(idx.value_counts(normalize=True), exp)