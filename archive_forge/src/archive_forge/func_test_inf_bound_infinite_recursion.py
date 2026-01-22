from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
@pytest.mark.xfail(not IS64, reason='GH 23440')
@pytest.mark.parametrize('left, right, expected', [([-np.inf, 1.0], [1.0, 2.0], 0.0), ([-np.inf, -2.0], [-2.0, -1.0], -2.0), ([-2.0, -1.0], [-1.0, np.inf], 0.0), ([1.0, 2.0], [2.0, np.inf], 2.0)])
def test_inf_bound_infinite_recursion(self, left, right, expected):
    tree = IntervalTree(left * 101, right * 101)
    result = tree.root.pivot
    assert result == expected