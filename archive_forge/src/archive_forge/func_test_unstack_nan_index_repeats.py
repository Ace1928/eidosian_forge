from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.parametrize('idx', itertools.permutations(['1st', '2nd', '3rd']))
@pytest.mark.parametrize('lev', list(range(3)))
@pytest.mark.parametrize('col', ['4th', '5th'])
def test_unstack_nan_index_repeats(self, idx, lev, col):

    def cast(val):
        val_str = '' if val != val else val
        return f'{val_str:1}'
    df = DataFrame({'1st': ['d'] * 3 + [np.nan] * 5 + ['a'] * 2 + ['c'] * 3 + ['e'] * 2 + ['b'] * 5, '2nd': ['y'] * 2 + ['w'] * 3 + [np.nan] * 3 + ['z'] * 4 + [np.nan] * 3 + ['x'] * 3 + [np.nan] * 2, '3rd': [67, 39, 53, 72, 57, 80, 31, 18, 11, 30, 59, 50, 62, 59, 76, 52, 14, 53, 60, 51]})
    df['4th'], df['5th'] = (df.apply(lambda r: '.'.join(map(cast, r)), axis=1), df.apply(lambda r: '.'.join(map(cast, r.iloc[::-1])), axis=1))
    mi = df.set_index(list(idx))
    udf = mi.unstack(level=lev)
    assert udf.notna().values.sum() == 2 * len(df)
    mk_list = lambda a: list(a) if isinstance(a, tuple) else [a]
    rows, cols = udf[col].notna().values.nonzero()
    for i, j in zip(rows, cols):
        left = sorted(udf[col].iloc[i, j].split('.'))
        right = mk_list(udf[col].index[i]) + mk_list(udf[col].columns[j])
        right = sorted(map(cast, right))
        assert left == right