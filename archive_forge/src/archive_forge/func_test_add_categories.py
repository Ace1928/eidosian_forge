import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_add_categories(self):
    cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
    old = cat.copy()
    new = Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c', 'd'], ordered=True)
    res = cat.add_categories('d')
    tm.assert_categorical_equal(cat, old)
    tm.assert_categorical_equal(res, new)
    res = cat.add_categories(['d'])
    tm.assert_categorical_equal(cat, old)
    tm.assert_categorical_equal(res, new)
    cat = Categorical(list('abc'), ordered=True)
    expected = Categorical(list('abc'), categories=list('abcde'), ordered=True)
    res = cat.add_categories(Series(['d', 'e']))
    tm.assert_categorical_equal(res, expected)
    res = cat.add_categories(np.array(['d', 'e']))
    tm.assert_categorical_equal(res, expected)
    res = cat.add_categories(Index(['d', 'e']))
    tm.assert_categorical_equal(res, expected)
    res = cat.add_categories(['d', 'e'])
    tm.assert_categorical_equal(res, expected)