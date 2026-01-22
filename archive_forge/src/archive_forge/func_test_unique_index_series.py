import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
def test_unique_index_series(self, ordered):
    dtype = CategoricalDtype([3, 2, 1], ordered=ordered)
    c = Categorical([3, 1, 2, 2, 1], dtype=dtype)
    exp = Categorical([3, 1, 2], dtype=dtype)
    tm.assert_categorical_equal(c.unique(), exp)
    tm.assert_index_equal(Index(c).unique(), Index(exp))
    tm.assert_categorical_equal(Series(c).unique(), exp)
    c = Categorical([1, 1, 2, 2], dtype=dtype)
    exp = Categorical([1, 2], dtype=dtype)
    tm.assert_categorical_equal(c.unique(), exp)
    tm.assert_index_equal(Index(c).unique(), Index(exp))
    tm.assert_categorical_equal(Series(c).unique(), exp)