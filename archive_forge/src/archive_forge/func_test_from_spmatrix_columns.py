import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('columns', [['a', 'b'], pd.MultiIndex.from_product([['A'], ['a', 'b']]), ['a', 'a']])
def test_from_spmatrix_columns(self, columns):
    sp_sparse = pytest.importorskip('scipy.sparse')
    dtype = SparseDtype('float64', 0.0)
    mat = sp_sparse.random(10, 2, density=0.5)
    result = pd.DataFrame.sparse.from_spmatrix(mat, columns=columns)
    expected = pd.DataFrame(mat.toarray(), columns=columns).astype(dtype)
    tm.assert_frame_equal(result, expected)