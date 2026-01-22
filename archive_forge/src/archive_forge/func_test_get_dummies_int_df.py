import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_get_dummies_int_df(self, dtype):
    data = DataFrame({'A': [1, 2, 1], 'B': Categorical(['a', 'b', 'a']), 'C': [1, 2, 1], 'D': [1.0, 2.0, 1.0]})
    columns = ['C', 'D', 'A_1', 'A_2', 'B_a', 'B_b']
    expected = DataFrame([[1, 1.0, 1, 0, 1, 0], [2, 2.0, 0, 1, 0, 1], [1, 1.0, 1, 0, 1, 0]], columns=columns)
    expected[columns[2:]] = expected[columns[2:]].astype(dtype)
    result = get_dummies(data, columns=['A', 'B'], dtype=dtype)
    tm.assert_frame_equal(result, expected)