import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [([1, 2, 3], [1, 2, 3]), (np.array([1, 2, 3]), np.array([1, 2, 3]))])
def test_assert_almost_equal_iterables(a, b):
    _assert_almost_equal_both(a, b)