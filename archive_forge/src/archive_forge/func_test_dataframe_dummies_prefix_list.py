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
def test_dataframe_dummies_prefix_list(self, df, sparse):
    prefixes = ['from_A', 'from_B']
    result = get_dummies(df, prefix=prefixes, sparse=sparse)
    expected = DataFrame({'C': [1, 2, 3], 'from_A_a': [True, False, True], 'from_A_b': [False, True, False], 'from_B_b': [True, True, False], 'from_B_c': [False, False, True]})
    expected[['C']] = df[['C']]
    cols = ['from_A_a', 'from_A_b', 'from_B_b', 'from_B_c']
    expected = expected[['C'] + cols]
    typ = SparseArray if sparse else Series
    expected[cols] = expected[cols].apply(lambda x: typ(x))
    tm.assert_frame_equal(result, expected)