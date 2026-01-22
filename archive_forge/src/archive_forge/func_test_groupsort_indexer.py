from datetime import datetime
from itertools import permutations
import numpy as np
from pandas._libs import algos as libalgos
import pandas._testing as tm
def test_groupsort_indexer():
    a = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)
    b = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)
    result = libalgos.groupsort_indexer(a, 1000)[0]
    expected = np.argsort(a, kind='mergesort')
    expected = expected.astype(np.intp)
    tm.assert_numpy_array_equal(result, expected)
    key = a * 1000 + b
    result = libalgos.groupsort_indexer(key, 1000000)[0]
    expected = np.lexsort((b, a))
    expected = expected.astype(np.intp)
    tm.assert_numpy_array_equal(result, expected)