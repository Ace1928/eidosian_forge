from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_2d_float32(self):
    arr = np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32)
    indexer = [0, 2, -1, 1, -1]
    result = algos.take_nd(arr, indexer, axis=0)
    expected = arr.take(indexer, axis=0)
    expected[[2, 4], :] = np.nan
    tm.assert_almost_equal(result, expected)
    result = algos.take_nd(arr, indexer, axis=1)
    expected = arr.take(indexer, axis=1)
    expected[:, [2, 4]] = np.nan
    tm.assert_almost_equal(result, expected)