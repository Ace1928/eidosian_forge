from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_reindex_values():
    values = [1, 2, 3, 4]
    indices = [1, 1, 2, 2]
    df = DataFrame({'group': ['Group1', 'Group2'] * 2, 'value': values}, index=indices)
    expected = Series(values, index=indices, name='value')

    def reindex_helper(x):
        return x.reindex(np.arange(x.index.min(), x.index.max() + 1))
    result = df.groupby('group', group_keys=False).value.apply(reindex_helper)
    tm.assert_series_equal(expected, result)