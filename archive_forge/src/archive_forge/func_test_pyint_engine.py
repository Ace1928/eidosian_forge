from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_pyint_engine():
    N = 5
    keys = [tuple(arr) for arr in [[0] * 10 * N, [1] * 10 * N, [2] * 10 * N, [np.nan] * N + [2] * 9 * N, [0] * N + [2] * 9 * N, [np.nan] * N + [2] * 8 * N + [0] * N]]
    for idx, key_value in enumerate(keys):
        index = MultiIndex.from_tuples(keys)
        assert index.get_loc(key_value) == idx
        expected = np.arange(idx + 1, dtype=np.intp)
        result = index.get_indexer([keys[i] for i in expected])
        tm.assert_numpy_array_equal(result, expected)
    idces = range(len(keys))
    expected = np.array([-1] + list(idces), dtype=np.intp)
    missing = tuple([0, 1] * 5 * N)
    result = index.get_indexer([missing] + [keys[i] for i in idces])
    tm.assert_numpy_array_equal(result, expected)