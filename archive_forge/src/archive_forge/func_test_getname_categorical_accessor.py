import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
@pytest.mark.parametrize('method', [lambda x: x.cat.set_categories([1, 2, 3]), lambda x: x.cat.reorder_categories([2, 3, 1], ordered=True), lambda x: x.cat.rename_categories([1, 2, 3]), lambda x: x.cat.remove_unused_categories(), lambda x: x.cat.remove_categories([2]), lambda x: x.cat.add_categories([4]), lambda x: x.cat.as_ordered(), lambda x: x.cat.as_unordered()])
def test_getname_categorical_accessor(self, method):
    ser = Series([1, 2, 3], name='A').astype('category')
    expected = 'A'
    result = method(ser).name
    assert result == expected