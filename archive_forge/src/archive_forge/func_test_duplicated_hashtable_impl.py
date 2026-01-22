from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_duplicated_hashtable_impl(keep, monkeypatch):
    n, k = (6, 10)
    levels = [np.arange(n), [str(i) for i in range(n)], 1000 + np.arange(n)]
    codes = [np.random.default_rng(2).choice(n, k * n) for _ in levels]
    with monkeypatch.context() as m:
        m.setattr(libindex, '_SIZE_CUTOFF', 50)
        mi = MultiIndex(levels=levels, codes=codes)
        result = mi.duplicated(keep=keep)
        expected = hashtable.duplicated(mi.values, keep=keep)
    tm.assert_numpy_array_equal(result, expected)