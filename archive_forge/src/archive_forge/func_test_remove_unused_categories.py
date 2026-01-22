import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_remove_unused_categories(self):
    c = Categorical(['a', 'b', 'c', 'd', 'a'], categories=['a', 'b', 'c', 'd', 'e'])
    exp_categories_all = Index(['a', 'b', 'c', 'd', 'e'])
    exp_categories_dropped = Index(['a', 'b', 'c', 'd'])
    tm.assert_index_equal(c.categories, exp_categories_all)
    res = c.remove_unused_categories()
    tm.assert_index_equal(res.categories, exp_categories_dropped)
    tm.assert_index_equal(c.categories, exp_categories_all)
    c = Categorical(['a', 'b', 'c', np.nan], categories=['a', 'b', 'c', 'd', 'e'])
    res = c.remove_unused_categories()
    tm.assert_index_equal(res.categories, Index(np.array(['a', 'b', 'c'])))
    exp_codes = np.array([0, 1, 2, -1], dtype=np.int8)
    tm.assert_numpy_array_equal(res.codes, exp_codes)
    tm.assert_index_equal(c.categories, exp_categories_all)
    val = ['F', np.nan, 'D', 'B', 'D', 'F', np.nan]
    cat = Categorical(values=val, categories=list('ABCDEFG'))
    out = cat.remove_unused_categories()
    tm.assert_index_equal(out.categories, Index(['B', 'D', 'F']))
    exp_codes = np.array([2, -1, 1, 0, 1, 2, -1], dtype=np.int8)
    tm.assert_numpy_array_equal(out.codes, exp_codes)
    assert out.tolist() == val
    alpha = list('abcdefghijklmnopqrstuvwxyz')
    val = np.random.default_rng(2).choice(alpha[::2], 10000).astype('object')
    val[np.random.default_rng(2).choice(len(val), 100)] = np.nan
    cat = Categorical(values=val, categories=alpha)
    out = cat.remove_unused_categories()
    assert out.tolist() == val.tolist()