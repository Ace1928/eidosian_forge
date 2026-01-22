from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_multi_iter_frame(self, three_group):
    k1 = np.array(['b', 'b', 'b', 'a', 'a', 'a'])
    k2 = np.array(['1', '2', '1', '2', '1', '2'])
    df = DataFrame({'v1': np.random.default_rng(2).standard_normal(6), 'v2': np.random.default_rng(2).standard_normal(6), 'k1': k1, 'k2': k2}, index=['one', 'two', 'three', 'four', 'five', 'six'])
    grouped = df.groupby(['k1', 'k2'])
    iterated = list(grouped)
    idx = df.index
    expected = [('a', '1', df.loc[idx[[4]]]), ('a', '2', df.loc[idx[[3, 5]]]), ('b', '1', df.loc[idx[[0, 2]]]), ('b', '2', df.loc[idx[[1]]])]
    for i, ((one, two), three) in enumerate(iterated):
        e1, e2, e3 = expected[i]
        assert e1 == one
        assert e2 == two
        tm.assert_frame_equal(three, e3)
    df['k1'] = np.array(['b', 'b', 'b', 'a', 'a', 'a'])
    df['k2'] = np.array(['1', '1', '1', '2', '2', '2'])
    grouped = df.groupby(['k1', 'k2'])
    groups = {key: gp for key, gp in grouped}
    assert len(groups) == 2
    three_levels = three_group.groupby(['A', 'B', 'C']).mean()
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        grouped = three_levels.T.groupby(axis=1, level=(1, 2))
    for key, group in grouped:
        pass