import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a1, a2, categories', [(['a', 'b', 'c'], [np.nan, 'a', 'b'], ['a', 'b', 'c']), ([1, 2, 3], [np.nan, 1, 2], [1, 2, 3])])
def test_compare_categorical_with_missing(self, a1, a2, categories):
    cat_type = CategoricalDtype(categories)
    result = Series(a1, dtype=cat_type) != Series(a2, dtype=cat_type)
    expected = Series(a1) != Series(a2)
    tm.assert_series_equal(result, expected)
    result = Series(a1, dtype=cat_type) == Series(a2, dtype=cat_type)
    expected = Series(a1) == Series(a2)
    tm.assert_series_equal(result, expected)