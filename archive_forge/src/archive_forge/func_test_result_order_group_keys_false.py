from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_result_order_group_keys_false():
    df = DataFrame({'A': [2, 1, 2], 'B': [1, 2, 3]})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A', group_keys=False).apply(lambda x: x)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = df.groupby('A', group_keys=False).apply(lambda x: x.copy())
    tm.assert_frame_equal(result, expected)