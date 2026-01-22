import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('vals', [[np.nan, np.nan, np.nan, np.nan, np.nan], [1, np.nan, np.nan, 3, np.nan], [1, np.nan, 0, 3, 0]])
@pytest.mark.parametrize('fill_value', [None, 0])
def test_dense_repr(self, vals, fill_value):
    vals = np.array(vals)
    arr = SparseArray(vals, fill_value=fill_value)
    res = arr.to_dense()
    tm.assert_numpy_array_equal(res, vals)