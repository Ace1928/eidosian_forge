import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_set_dtype_new_categories(self):
    c = Categorical(['a', 'b', 'c'])
    result = c._set_dtype(CategoricalDtype(list('abcd')))
    tm.assert_numpy_array_equal(result.codes, c.codes)
    tm.assert_index_equal(result.dtype.categories, Index(list('abcd')))