from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('f', [lambda x: x, lambda x: Series(x), lambda x: x.values])
def test_from_product_index_series_categorical(ordered, f):
    first = ['foo', 'bar']
    idx = pd.CategoricalIndex(list('abcaab'), categories=list('bac'), ordered=ordered)
    expected = pd.CategoricalIndex(list('abcaab') + list('abcaab'), categories=list('bac'), ordered=ordered)
    result = MultiIndex.from_product([first, f(idx)])
    tm.assert_index_equal(result.get_level_values(1), expected)