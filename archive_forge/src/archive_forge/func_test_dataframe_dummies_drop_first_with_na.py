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
def test_dataframe_dummies_drop_first_with_na(self, df, sparse):
    df.loc[3, :] = [np.nan, np.nan, np.nan]
    result = get_dummies(df, dummy_na=True, drop_first=True, sparse=sparse).sort_index(axis=1)
    expected = DataFrame({'C': [1, 2, 3, np.nan], 'A_b': [0, 1, 0, 0], 'A_nan': [0, 0, 0, 1], 'B_c': [0, 0, 1, 0], 'B_nan': [0, 0, 0, 1]})
    cols = ['A_b', 'A_nan', 'B_c', 'B_nan']
    expected[cols] = expected[cols].astype(bool)
    expected = expected.sort_index(axis=1)
    if sparse:
        for col in cols:
            expected[col] = SparseArray(expected[col])
    tm.assert_frame_equal(result, expected)
    result = get_dummies(df, dummy_na=False, drop_first=True, sparse=sparse)
    expected = expected[['C', 'A_b', 'B_c']]
    tm.assert_frame_equal(result, expected)