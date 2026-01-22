import numpy as np
import pytest
from pandas import CategoricalIndex
import pandas._testing as tm
def test_fillna_categorical(self):
    idx = CategoricalIndex([1.0, np.nan, 3.0, 1.0], name='x')
    exp = CategoricalIndex([1.0, 1.0, 3.0, 1.0], name='x')
    tm.assert_index_equal(idx.fillna(1.0), exp)
    cat = idx._data
    msg = 'Cannot setitem on a Categorical with a new category'
    with pytest.raises(TypeError, match=msg):
        cat.fillna(2.0)
    result = idx.fillna(2.0)
    expected = idx.astype(object).fillna(2.0)
    tm.assert_index_equal(result, expected)