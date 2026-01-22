from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('n', range(1, 6))
@pytest.mark.parametrize('m', range(1, 5))
def test_duplicated_with_nan_multi_shape(n, m):
    codes = product(range(-1, n), range(-1, m))
    mi = MultiIndex(levels=[list('abcde')[:n], list('WXYZ')[:m]], codes=np.random.default_rng(2).permutation(list(codes)).T)
    assert len(mi) == (n + 1) * (m + 1)
    assert not mi.has_duplicates
    tm.assert_numpy_array_equal(mi.duplicated(), np.zeros(len(mi), dtype='bool'))