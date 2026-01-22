from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_func_that_appends_group_to_list_without_copy():
    df = DataFrame(1, index=list(range(10)) * 10, columns=[0]).reset_index()
    groups = []

    def store(group):
        groups.append(group)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        df.groupby('index').apply(store)
    expected_value = DataFrame({'index': [0] * 10, 0: [1] * 10}, index=pd.RangeIndex(0, 100, 10))
    tm.assert_frame_equal(groups[0], expected_value)