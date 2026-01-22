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
def test_get_dummies_include_na(self, sparse, dtype):
    s = ['a', 'b', np.nan]
    res = get_dummies(s, sparse=sparse, dtype=dtype)
    exp = DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0]}, dtype=self.effective_dtype(dtype))
    if sparse:
        if dtype.kind == 'b':
            exp = exp.apply(SparseArray, fill_value=False)
        else:
            exp = exp.apply(SparseArray, fill_value=0.0)
    tm.assert_frame_equal(res, exp)
    res_na = get_dummies(s, dummy_na=True, sparse=sparse, dtype=dtype)
    exp_na = DataFrame({np.nan: [0, 0, 1], 'a': [1, 0, 0], 'b': [0, 1, 0]}, dtype=self.effective_dtype(dtype))
    exp_na = exp_na.reindex(['a', 'b', np.nan], axis=1)
    exp_na.columns = res_na.columns
    if sparse:
        if dtype.kind == 'b':
            exp_na = exp_na.apply(SparseArray, fill_value=False)
        else:
            exp_na = exp_na.apply(SparseArray, fill_value=0.0)
    tm.assert_frame_equal(res_na, exp_na)
    res_just_na = get_dummies([np.nan], dummy_na=True, sparse=sparse, dtype=dtype)
    exp_just_na = DataFrame(Series(1, index=[0]), columns=[np.nan], dtype=self.effective_dtype(dtype))
    tm.assert_numpy_array_equal(res_just_na.values, exp_just_na.values)