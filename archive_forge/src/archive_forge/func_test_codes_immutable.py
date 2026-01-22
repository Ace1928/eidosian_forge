import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_codes_immutable(self):
    c = Categorical(['a', 'b', 'c', 'a', np.nan])
    exp = np.array([0, 1, 2, 0, -1], dtype='int8')
    tm.assert_numpy_array_equal(c.codes, exp)
    msg = "property 'codes' of 'Categorical' object has no setter" if PY311 else "can't set attribute"
    with pytest.raises(AttributeError, match=msg):
        c.codes = np.array([0, 1, 2, 0, 1], dtype='int8')
    codes = c.codes
    with pytest.raises(ValueError, match='assignment destination is read-only'):
        codes[4] = 1
    c[4] = 'a'
    exp = np.array([0, 1, 2, 0, 0], dtype='int8')
    tm.assert_numpy_array_equal(c.codes, exp)
    c._codes[4] = 2
    exp = np.array([0, 1, 2, 0, 2], dtype='int8')
    tm.assert_numpy_array_equal(c.codes, exp)