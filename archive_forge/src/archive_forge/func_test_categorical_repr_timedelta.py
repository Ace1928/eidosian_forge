import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_repr_timedelta(self):
    idx = timedelta_range('1 days', periods=5)
    c = Categorical(idx)
    exp = '[1 days, 2 days, 3 days, 4 days, 5 days]\nCategories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]'
    assert repr(c) == exp
    c = Categorical(idx.append(idx), categories=idx)
    exp = '[1 days, 2 days, 3 days, 4 days, 5 days, 1 days, 2 days, 3 days, 4 days, 5 days]\nCategories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]'
    assert repr(c) == exp
    idx = timedelta_range('1 hours', periods=20)
    c = Categorical(idx)
    exp = '[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]\nLength: 20\nCategories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,\n                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,\n                                   18 days 01:00:00, 19 days 01:00:00]'
    assert repr(c) == exp
    c = Categorical(idx.append(idx), categories=idx)
    exp = '[0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00, 3 days 01:00:00, 4 days 01:00:00, ..., 15 days 01:00:00, 16 days 01:00:00, 17 days 01:00:00, 18 days 01:00:00, 19 days 01:00:00]\nLength: 40\nCategories (20, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,\n                                   3 days 01:00:00, ..., 16 days 01:00:00, 17 days 01:00:00,\n                                   18 days 01:00:00, 19 days 01:00:00]'
    assert repr(c) == exp