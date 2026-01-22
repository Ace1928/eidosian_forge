from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_multiindex_partial_indexing_equivalence(self):
    df = DataFrame([[1, 2, 3, 4], [3, 4, 5, 6], [1, 4, 2, 3]], columns=MultiIndex.from_arrays([['a', 'b', 'b', 'c'], [1, 1, 2, 2]]))
    expected_mean = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].mean()
    result_mean = df.groupby([('a', 1)])['b'].mean()
    tm.assert_frame_equal(expected_mean, result_mean)
    expected_sum = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].sum()
    result_sum = df.groupby([('a', 1)])['b'].sum()
    tm.assert_frame_equal(expected_sum, result_sum)
    expected_count = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].count()
    result_count = df.groupby([('a', 1)])['b'].count()
    tm.assert_frame_equal(expected_count, result_count)
    expected_min = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].min()
    result_min = df.groupby([('a', 1)])['b'].min()
    tm.assert_frame_equal(expected_min, result_min)
    expected_max = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].max()
    result_max = df.groupby([('a', 1)])['b'].max()
    tm.assert_frame_equal(expected_max, result_max)
    expected_groups = df.groupby([('a', 1)])[[('b', 1), ('b', 2)]].groups
    result_groups = df.groupby([('a', 1)])['b'].groups
    tm.assert_dict_equal(expected_groups, result_groups)