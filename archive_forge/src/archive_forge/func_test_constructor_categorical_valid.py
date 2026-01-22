from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
@pytest.mark.parametrize('cat_constructor', [Categorical, CategoricalIndex])
def test_constructor_categorical_valid(self, constructor, cat_constructor):
    breaks = np.arange(10, dtype='int64')
    expected = IntervalIndex.from_breaks(breaks)
    cat_breaks = cat_constructor(breaks)
    result_kwargs = self.get_kwargs_from_breaks(cat_breaks)
    result = constructor(**result_kwargs)
    tm.assert_index_equal(result, expected)