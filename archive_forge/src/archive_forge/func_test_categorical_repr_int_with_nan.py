import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_repr_int_with_nan(self):
    c = Categorical([1, 2, np.nan])
    c_exp = '[1, 2, NaN]\nCategories (2, int64): [1, 2]'
    assert repr(c) == c_exp
    s = Series([1, 2, np.nan], dtype='object').astype('category')
    s_exp = '0      1\n1      2\n2    NaN\ndtype: category\nCategories (2, int64): [1, 2]'
    assert repr(s) == s_exp