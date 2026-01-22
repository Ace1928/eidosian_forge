from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('index2,expected_arr', [(Index(['B', 'D']), ['B']), (Index(['B', 'D', 'A']), ['A', 'B'])])
def test_intersection_non_monotonic_non_unique(self, index2, expected_arr, sort):
    index1 = Index(['A', 'B', 'A', 'C'])
    expected = Index(expected_arr)
    result = index1.intersection(index2, sort=sort)
    if sort is None:
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)