from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@pytest.mark.parametrize('shuffle', [True, False], ids=['unaligned', 'aligned'])
@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_series(flip, shuffle, sparse, ufunc, arrays_for_binary_ufunc):
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name = 'name'
    series = pd.Series(a1, name=name)
    other = pd.Series(a2, name=name)
    idx = np.random.default_rng(2).permutation(len(a1))
    if shuffle:
        other = other.take(idx)
        if flip:
            index = other.align(series)[0].index
        else:
            index = series.align(other)[0].index
    else:
        index = series.index
    array_args = (a1, a2)
    series_args = (series, other)
    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))
    expected = pd.Series(ufunc(*array_args), index=index, name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)