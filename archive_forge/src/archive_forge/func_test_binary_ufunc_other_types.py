from collections import deque
import re
import string
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
@pytest.mark.parametrize('type_', [list, deque, tuple])
def test_binary_ufunc_other_types(type_):
    a = pd.Series([1, 2, 3], name='name')
    b = type_([3, 4, 5])
    result = np.add(a, b)
    expected = pd.Series(np.add(a.to_numpy(), b), name='name')
    tm.assert_series_equal(result, expected)