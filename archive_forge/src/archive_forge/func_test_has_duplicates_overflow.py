from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('nlevels', [4, 8])
@pytest.mark.parametrize('with_nulls', [True, False])
def test_has_duplicates_overflow(nlevels, with_nulls):
    codes = np.tile(np.arange(500), 2)
    level = np.arange(500)
    if with_nulls:
        codes[500] = -1
        codes = [codes.copy() for i in range(nlevels)]
        for i in range(nlevels):
            codes[i][500 + i - nlevels // 2] = -1
        codes += [np.array([-1, 1]).repeat(500)]
    else:
        codes = [codes] * nlevels + [np.arange(2).repeat(500)]
    levels = [level] * nlevels + [[0, 1]]
    mi = MultiIndex(levels=levels, codes=codes)
    assert not mi.has_duplicates
    if with_nulls:

        def f(a):
            return np.insert(a, 1000, a[0])
        codes = list(map(f, codes))
        mi = MultiIndex(levels=levels, codes=codes)
    else:
        values = mi.values.tolist()
        mi = MultiIndex.from_tuples(values + [values[0]])
    assert mi.has_duplicates