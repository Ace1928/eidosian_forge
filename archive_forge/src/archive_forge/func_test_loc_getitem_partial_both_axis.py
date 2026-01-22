import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_loc_getitem_partial_both_axis():
    iterables = [['a', 'b'], [2, 1]]
    columns = MultiIndex.from_product(iterables, names=['col1', 'col2'])
    rows = MultiIndex.from_product(iterables, names=['row1', 'row2'])
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=rows, columns=columns)
    expected = df.iloc[:2, 2:].droplevel('row1').droplevel('col1', axis=1)
    result = df.loc['a', 'b']
    tm.assert_frame_equal(result, expected)