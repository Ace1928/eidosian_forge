from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_series_groupby_value_counts_on_categorical():
    s = Series(Categorical(['a'], categories=['a', 'b']))
    result = s.groupby([0]).value_counts()
    expected = Series(data=[1, 0], index=MultiIndex.from_arrays([np.array([0, 0]), CategoricalIndex(['a', 'b'], categories=['a', 'b'], ordered=False, dtype='category')]), name='count')
    tm.assert_series_equal(result, expected)