from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_with_key_not_unique(self):
    df1 = DataFrame({'name': [1]})
    df2 = DataFrame({'name': [2]})
    df3 = DataFrame({'name': [3]})
    df_a = concat([df1, df2, df3], keys=['x', 'y', 'x'])
    with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
        out_a = df_a.loc[('x', 0), :]
    df_b = DataFrame({'name': [1, 2, 3]}, index=Index([('x', 0), ('y', 0), ('x', 0)]))
    with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
        out_b = df_b.loc['x', 0]
    tm.assert_frame_equal(out_a, out_b)
    df1 = DataFrame({'name': ['a', 'a', 'b']})
    df2 = DataFrame({'name': ['a', 'b']})
    df3 = DataFrame({'name': ['c', 'd']})
    df_a = concat([df1, df2, df3], keys=['x', 'y', 'x'])
    with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
        out_a = df_a.loc[('x', 0), :]
    df_b = DataFrame({'a': ['x', 'x', 'x', 'y', 'y', 'x', 'x'], 'b': [0, 1, 2, 0, 1, 0, 1], 'name': list('aababcd')}).set_index(['a', 'b'])
    df_b.index.names = [None, None]
    with tm.assert_produces_warning(PerformanceWarning, match='indexing past lexsort depth'):
        out_b = df_b.loc[('x', 0), :]
    tm.assert_frame_equal(out_a, out_b)