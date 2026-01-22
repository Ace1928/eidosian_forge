from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_positional_slice_groups_datetimelike():
    expected = DataFrame({'date': pd.date_range('2010-01-01', freq='12h', periods=5), 'vals': range(5), 'let': list('abcde')})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = expected.groupby([expected.let, expected.date.dt.date], group_keys=False).apply(lambda x: x.iloc[0:])
    tm.assert_frame_equal(result, expected)