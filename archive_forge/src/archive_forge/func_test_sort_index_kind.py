import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_index_kind(self, sort_kind):
    series = Series(index=[3, 2, 1, 4, 3], dtype=object)
    expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)
    index_sorted_series = series.sort_index(kind=sort_kind)
    tm.assert_series_equal(expected_series, index_sorted_series)