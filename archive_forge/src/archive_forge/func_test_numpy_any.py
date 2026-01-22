import numpy as np
import pytest
from pandas import (
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('data,pos,neg', [([False, True, False], True, False), ([0, 2, 0], 2, 0), ([0.0, 2.0, 0.0], 2.0, 0.0)])
def test_numpy_any(self, data, pos, neg):
    out = np.any(SparseArray(data))
    assert out
    out = np.any(SparseArray(data, fill_value=pos))
    assert out
    data[1] = neg
    out = np.any(SparseArray(data))
    assert not out
    out = np.any(SparseArray(data, fill_value=pos))
    assert not out
    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.any(SparseArray(data), out=out)