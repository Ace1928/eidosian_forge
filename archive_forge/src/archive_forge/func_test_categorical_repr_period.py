import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_repr_period(self):
    idx = period_range('2011-01-01 09:00', freq='h', periods=5)
    c = Categorical(idx)
    exp = '[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]\nCategories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,\n                            2011-01-01 13:00]'
    assert repr(c) == exp
    c = Categorical(idx.append(idx), categories=idx)
    exp = '[2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00, 2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00, 2011-01-01 13:00]\nCategories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,\n                            2011-01-01 13:00]'
    assert repr(c) == exp
    idx = period_range('2011-01', freq='M', periods=5)
    c = Categorical(idx)
    exp = '[2011-01, 2011-02, 2011-03, 2011-04, 2011-05]\nCategories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]'
    assert repr(c) == exp
    c = Categorical(idx.append(idx), categories=idx)
    exp = '[2011-01, 2011-02, 2011-03, 2011-04, 2011-05, 2011-01, 2011-02, 2011-03, 2011-04, 2011-05]\nCategories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]'
    assert repr(c) == exp