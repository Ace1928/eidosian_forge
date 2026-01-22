import numpy as np
import pytest
from pandas._libs import lib
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('binner,closed,expected', [(np.array([0, 3, 6, 9], dtype=np.int64), 'left', np.array([2, 5, 6], dtype=np.int64)), (np.array([0, 3, 6, 9], dtype=np.int64), 'right', np.array([3, 6, 6], dtype=np.int64)), (np.array([0, 3, 6], dtype=np.int64), 'left', np.array([2, 5], dtype=np.int64)), (np.array([0, 3, 6], dtype=np.int64), 'right', np.array([3, 6], dtype=np.int64))])
def test_generate_bins(binner, closed, expected):
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = lib.generate_bins_dt64(values, binner, closed=closed)
    tm.assert_numpy_array_equal(result, expected)