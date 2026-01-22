from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('group_keys', [True, False])
def test_apply_chunk_view(group_keys):
    df = DataFrame({'key': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'value': range(9)})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('key', group_keys=group_keys).apply(lambda x: x.iloc[:2])
    expected = df.take([0, 1, 3, 4, 6, 7])
    if group_keys:
        expected.index = MultiIndex.from_arrays([[1, 1, 2, 2, 3, 3], expected.index], names=['key', None])
    tm.assert_frame_equal(result, expected)