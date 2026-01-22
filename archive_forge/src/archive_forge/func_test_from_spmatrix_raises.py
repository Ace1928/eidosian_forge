import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_from_spmatrix_raises(self):
    sp_sparse = pytest.importorskip('scipy.sparse')
    mat = sp_sparse.eye(5, 4, format='csc')
    with pytest.raises(ValueError, match="not '4'"):
        SparseArray.from_spmatrix(mat)