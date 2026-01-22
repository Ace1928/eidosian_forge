import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_index_kind_neg_key(self, sort_kind):
    series = Series(index=[3, 2, 1, 4, 3], dtype=object)
    expected_series = Series(index=[4, 3, 3, 2, 1], dtype=object)
    index_sorted_series = series.sort_index(kind=sort_kind, key=lambda x: -x)
    tm.assert_series_equal(expected_series, index_sorted_series)