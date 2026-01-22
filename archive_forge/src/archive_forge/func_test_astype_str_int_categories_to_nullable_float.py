import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_str_int_categories_to_nullable_float(self):
    dtype = CategoricalDtype([str(i / 2) for i in range(5)])
    codes = np.random.default_rng(2).integers(5, size=20)
    arr = Categorical.from_codes(codes, dtype=dtype)
    res = arr.astype('Float64')
    expected = array(codes, dtype='Float64') / 2
    tm.assert_extension_array_equal(res, expected)