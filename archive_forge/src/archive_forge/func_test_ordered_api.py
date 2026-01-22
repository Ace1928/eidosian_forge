import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_ordered_api(self):
    cat1 = Categorical(list('acb'), ordered=False)
    tm.assert_index_equal(cat1.categories, Index(['a', 'b', 'c']))
    assert not cat1.ordered
    cat2 = Categorical(list('acb'), categories=list('bca'), ordered=False)
    tm.assert_index_equal(cat2.categories, Index(['b', 'c', 'a']))
    assert not cat2.ordered
    cat3 = Categorical(list('acb'), ordered=True)
    tm.assert_index_equal(cat3.categories, Index(['a', 'b', 'c']))
    assert cat3.ordered
    cat4 = Categorical(list('acb'), categories=list('bca'), ordered=True)
    tm.assert_index_equal(cat4.categories, Index(['b', 'c', 'a']))
    assert cat4.ordered