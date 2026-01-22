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
def test_dataframe_dummies_string_dtype(self, df, using_infer_string):
    df = df[['A', 'B']]
    df = df.astype({'A': 'object', 'B': 'string'})
    result = get_dummies(df)
    expected = DataFrame({'A_a': [1, 0, 1], 'A_b': [0, 1, 0], 'B_b': [1, 1, 0], 'B_c': [0, 0, 1]}, dtype=bool)
    if not using_infer_string:
        expected[['B_b', 'B_c']] = expected[['B_b', 'B_c']].astype('boolean')
    tm.assert_frame_equal(result, expected)