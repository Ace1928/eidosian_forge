from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_multiindex_tuple(self):
    df = DataFrame([[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]], columns=MultiIndex.from_arrays([['a', 'b', 'b', 'c'], [1, 1, 2, 2]]))
    expected = df.groupby([('b', 1)]).groups
    result = df.groupby(('b', 1)).groups
    tm.assert_dict_equal(expected, result)
    df2 = DataFrame(df.values, columns=MultiIndex.from_arrays([['a', 'b', 'b', 'c'], ['d', 'd', 'e', 'e']]))
    expected = df2.groupby([('b', 'd')]).groups
    result = df.groupby(('b', 1)).groups
    tm.assert_dict_equal(expected, result)
    df3 = DataFrame(df.values, columns=[('a', 'd'), ('b', 'd'), ('b', 'e'), 'c'])
    expected = df3.groupby([('b', 'd')]).groups
    result = df.groupby(('b', 1)).groups
    tm.assert_dict_equal(expected, result)