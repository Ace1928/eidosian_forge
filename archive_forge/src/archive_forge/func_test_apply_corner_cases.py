from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_corner_cases():
    N = 1000
    labels = np.random.default_rng(2).integers(0, 100, size=N)
    df = DataFrame({'key': labels, 'value1': np.random.default_rng(2).standard_normal(N), 'value2': ['foo', 'bar', 'baz', 'qux'] * (N // 4)})
    grouped = df.groupby('key', group_keys=False)

    def f(g):
        g['value3'] = g['value1'] * 2
        return g
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = grouped.apply(f)
    assert 'value3' in result