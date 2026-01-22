import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_construction_with_dtype(self):
    ci = CategoricalIndex(list('aabbca'), categories=list('abc'), ordered=False)
    result = Index(np.array(ci), dtype='category')
    tm.assert_index_equal(result, ci, exact=True)
    result = Index(np.array(ci).tolist(), dtype='category')
    tm.assert_index_equal(result, ci, exact=True)
    ci = CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=False)
    result = Index(np.array(ci), dtype='category').reorder_categories(ci.categories)
    tm.assert_index_equal(result, ci, exact=True)
    idx = Index(range(3))
    expected = CategoricalIndex([0, 1, 2], categories=idx, ordered=True)
    result = CategoricalIndex(idx, categories=idx, ordered=True)
    tm.assert_index_equal(result, expected, exact=True)