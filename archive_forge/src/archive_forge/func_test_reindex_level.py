import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_level(idx):
    index = Index(['one'])
    target, indexer = idx.reindex(index, level='second')
    target2, indexer2 = index.reindex(idx, level='second')
    exp_index = idx.join(index, level='second', how='right')
    exp_index2 = idx.join(index, level='second', how='left')
    assert target.equals(exp_index)
    exp_indexer = np.array([0, 2, 4])
    tm.assert_numpy_array_equal(indexer, exp_indexer, check_dtype=False)
    assert target2.equals(exp_index2)
    exp_indexer2 = np.array([0, -1, 0, -1, 0, -1])
    tm.assert_numpy_array_equal(indexer2, exp_indexer2, check_dtype=False)
    with pytest.raises(TypeError, match='Fill method not supported'):
        idx.reindex(idx, method='pad', level='second')