import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_repr_ordered(self):
    c = Categorical([1, 2, 3], ordered=True)
    exp = '[1, 2, 3]\nCategories (3, int64): [1 < 2 < 3]'
    assert repr(c) == exp
    c = Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2, 3], ordered=True)
    exp = '[1, 2, 3, 1, 2, 3]\nCategories (3, int64): [1 < 2 < 3]'
    assert repr(c) == exp
    c = Categorical([1, 2, 3, 4, 5] * 10, ordered=True)
    exp = '[1, 2, 3, 4, 5, ..., 1, 2, 3, 4, 5]\nLength: 50\nCategories (5, int64): [1 < 2 < 3 < 4 < 5]'
    assert repr(c) == exp
    c = Categorical(np.arange(20, dtype=np.int64), ordered=True)
    exp = '[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]\nLength: 20\nCategories (20, int64): [0 < 1 < 2 < 3 ... 16 < 17 < 18 < 19]'
    assert repr(c) == exp