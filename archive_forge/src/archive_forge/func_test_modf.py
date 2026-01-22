import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('fill_value', [0.0, np.nan])
def test_modf(self, fill_value):
    sparse = SparseArray([fill_value] * 10 + [1.1, 2.2], fill_value=fill_value)
    r1, r2 = np.modf(sparse)
    e1, e2 = np.modf(np.asarray(sparse))
    tm.assert_sp_array_equal(r1, SparseArray(e1, fill_value=fill_value))
    tm.assert_sp_array_equal(r2, SparseArray(e2, fill_value=fill_value))