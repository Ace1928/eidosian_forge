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
def test_dataframe_dummies_prefix_sep(self, df, sparse):
    result = get_dummies(df, prefix_sep='..', sparse=sparse)
    expected = DataFrame({'C': [1, 2, 3], 'A..a': [True, False, True], 'A..b': [False, True, False], 'B..b': [True, True, False], 'B..c': [False, False, True]})
    expected[['C']] = df[['C']]
    expected = expected[['C', 'A..a', 'A..b', 'B..b', 'B..c']]
    if sparse:
        cols = ['A..a', 'A..b', 'B..b', 'B..c']
        expected[cols] = expected[cols].astype(SparseDtype('bool', False))
    tm.assert_frame_equal(result, expected)
    result = get_dummies(df, prefix_sep=['..', '__'], sparse=sparse)
    expected = expected.rename(columns={'B..b': 'B__b', 'B..c': 'B__c'})
    tm.assert_frame_equal(result, expected)
    result = get_dummies(df, prefix_sep={'A': '..', 'B': '__'}, sparse=sparse)
    tm.assert_frame_equal(result, expected)