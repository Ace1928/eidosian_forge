import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_update_with_categorical_type(self):
    dtype = CategoricalDtype(['a', 'b', 'c', 'd'])
    s1 = Series(['a', 'b', 'c'], index=[1, 2, 3], dtype=dtype)
    s2 = Series(['b', 'a'], index=[1, 2], dtype=dtype)
    s1.update(s2)
    result = s1
    expected = Series(['b', 'a', 'c'], index=[1, 2, 3], dtype=dtype)
    tm.assert_series_equal(result, expected)