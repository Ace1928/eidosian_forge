import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_categorical_multiindex(self):
    df = DataFrame({'a': range(6), 'l1': pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'], categories=['c', 'a', 'b'], ordered=True), 'l2': [0, 1, 0, 1, 0, 1]})
    result = df.set_index(['l1', 'l2']).sort_index()
    expected = DataFrame([4, 5, 0, 1, 2, 3], columns=['a'], index=MultiIndex(levels=[CategoricalIndex(['c', 'a', 'b'], categories=['c', 'a', 'b'], ordered=True, name='l1', dtype='category'), [0, 1]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], names=['l1', 'l2']))
    tm.assert_frame_equal(result, expected)