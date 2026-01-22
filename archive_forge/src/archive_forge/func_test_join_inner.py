import numpy as np
from pandas import (
import pandas._testing as tm
def test_join_inner(self):
    index = RangeIndex(start=0, stop=20, step=2)
    other = Index(np.arange(25, 14, -1, dtype=np.int64))
    res, lidx, ridx = index.join(other, how='inner', return_indexers=True)
    ind = res.argsort()
    res = res.take(ind)
    lidx = lidx.take(ind)
    ridx = ridx.take(ind)
    eres = Index([16, 18])
    elidx = np.array([8, 9], dtype=np.intp)
    eridx = np.array([9, 7], dtype=np.intp)
    assert isinstance(res, Index) and res.dtype == np.int64
    tm.assert_index_equal(res, eres)
    tm.assert_numpy_array_equal(lidx, elidx)
    tm.assert_numpy_array_equal(ridx, eridx)
    other = RangeIndex(25, 14, -1)
    res, lidx, ridx = index.join(other, how='inner', return_indexers=True)
    assert isinstance(res, RangeIndex)
    tm.assert_index_equal(res, eres, exact='equiv')
    tm.assert_numpy_array_equal(lidx, elidx)
    tm.assert_numpy_array_equal(ridx, eridx)