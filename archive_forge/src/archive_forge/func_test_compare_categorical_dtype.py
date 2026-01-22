from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('reverse', [True, False])
@pytest.mark.parametrize('as_index', [True, False])
def test_compare_categorical_dtype(self, arr1d, as_index, reverse, ordered):
    other = pd.Categorical(arr1d, ordered=ordered)
    if as_index:
        other = pd.CategoricalIndex(other)
    left, right = (arr1d, other)
    if reverse:
        left, right = (right, left)
    ones = np.ones(arr1d.shape, dtype=bool)
    zeros = ~ones
    result = left == right
    tm.assert_numpy_array_equal(result, ones)
    result = left != right
    tm.assert_numpy_array_equal(result, zeros)
    if not reverse and (not as_index):
        result = left < right
        tm.assert_numpy_array_equal(result, zeros)
        result = left <= right
        tm.assert_numpy_array_equal(result, ones)
        result = left > right
        tm.assert_numpy_array_equal(result, zeros)
        result = left >= right
        tm.assert_numpy_array_equal(result, ones)