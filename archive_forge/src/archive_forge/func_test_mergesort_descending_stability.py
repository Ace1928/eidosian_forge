import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_mergesort_descending_stability(self):
    s = Series([1, 2, 1, 3], ['first', 'b', 'second', 'c'])
    result = s.sort_values(ascending=False, kind='mergesort')
    expected = Series([3, 2, 1, 1], ['c', 'b', 'first', 'second'])
    tm.assert_series_equal(result, expected)