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
def test_dataframe_dummies_drop_first_with_categorical(self, df, sparse, dtype):
    df['cat'] = Categorical(['x', 'y', 'y'])
    result = get_dummies(df, drop_first=True, sparse=sparse)
    expected = DataFrame({'C': [1, 2, 3], 'A_b': [0, 1, 0], 'B_c': [0, 0, 1], 'cat_y': [0, 1, 1]})
    cols = ['A_b', 'B_c', 'cat_y']
    expected[cols] = expected[cols].astype(bool)
    expected = expected[['C', 'A_b', 'B_c', 'cat_y']]
    if sparse:
        for col in cols:
            expected[col] = SparseArray(expected[col])
    tm.assert_frame_equal(result, expected)