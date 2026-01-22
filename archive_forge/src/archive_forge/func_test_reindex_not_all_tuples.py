import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_not_all_tuples():
    keys = [('i', 'i'), ('i', 'j'), ('j', 'i'), 'j']
    mi = MultiIndex.from_tuples(keys[:-1])
    idx = Index(keys)
    res, indexer = mi.reindex(idx)
    tm.assert_index_equal(res, idx)
    expected = np.array([0, 1, 2, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)