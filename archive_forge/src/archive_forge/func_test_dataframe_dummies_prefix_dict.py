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
def test_dataframe_dummies_prefix_dict(self, sparse):
    prefixes = {'A': 'from_A', 'B': 'from_B'}
    df = DataFrame({'C': [1, 2, 3], 'A': ['a', 'b', 'a'], 'B': ['b', 'b', 'c']})
    result = get_dummies(df, prefix=prefixes, sparse=sparse)
    expected = DataFrame({'C': [1, 2, 3], 'from_A_a': [1, 0, 1], 'from_A_b': [0, 1, 0], 'from_B_b': [1, 1, 0], 'from_B_c': [0, 0, 1]})
    columns = ['from_A_a', 'from_A_b', 'from_B_b', 'from_B_c']
    expected[columns] = expected[columns].astype(bool)
    if sparse:
        expected[columns] = expected[columns].astype(SparseDtype('bool', False))
    tm.assert_frame_equal(result, expected)