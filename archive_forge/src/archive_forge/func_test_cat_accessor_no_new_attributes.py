import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
def test_cat_accessor_no_new_attributes(self):
    cat = Series(list('aabbcde')).astype('category')
    with pytest.raises(AttributeError, match='You cannot add any new attribute'):
        cat.cat.xlabel = 'a'