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
def test_get_dummies_basic_drop_first_NA(self, sparse):
    s_NA = ['a', 'b', np.nan]
    res = get_dummies(s_NA, drop_first=True, sparse=sparse)
    exp = DataFrame({'b': [0, 1, 0]}, dtype=bool)
    if sparse:
        exp = exp.apply(SparseArray, fill_value=False)
    tm.assert_frame_equal(res, exp)
    res_na = get_dummies(s_NA, dummy_na=True, drop_first=True, sparse=sparse)
    exp_na = DataFrame({'b': [0, 1, 0], np.nan: [0, 0, 1]}, dtype=bool).reindex(['b', np.nan], axis=1)
    if sparse:
        exp_na = exp_na.apply(SparseArray, fill_value=False)
    tm.assert_frame_equal(res_na, exp_na)
    res_just_na = get_dummies([np.nan], dummy_na=True, drop_first=True, sparse=sparse)
    exp_just_na = DataFrame(index=RangeIndex(1))
    tm.assert_frame_equal(res_just_na, exp_just_na)