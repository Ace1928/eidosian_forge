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
def test_dataframe_dummies_drop_first(self, df, sparse):
    df = df[['A', 'B']]
    result = get_dummies(df, drop_first=True, sparse=sparse)
    expected = DataFrame({'A_b': [0, 1, 0], 'B_c': [0, 0, 1]}, dtype=bool)
    if sparse:
        expected = expected.apply(SparseArray, fill_value=False)
    tm.assert_frame_equal(result, expected)