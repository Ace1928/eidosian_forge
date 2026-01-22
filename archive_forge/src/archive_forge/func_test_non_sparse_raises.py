import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_non_sparse_raises(self):
    ser = pd.Series([1, 2, 3])
    with pytest.raises(AttributeError, match='.sparse'):
        ser.sparse.density