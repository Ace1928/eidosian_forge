import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_big_print(self):
    codes = np.array([0, 1, 2, 0, 1, 2] * 100)
    dtype = CategoricalDtype(categories=Index(['a', 'b', 'c'], dtype=object))
    factor = Categorical.from_codes(codes, dtype=dtype)
    expected = ["['a', 'b', 'c', 'a', 'b', ..., 'b', 'c', 'a', 'b', 'c']", 'Length: 600', "Categories (3, object): ['a', 'b', 'c']"]
    expected = '\n'.join(expected)
    actual = repr(factor)
    assert actual == expected