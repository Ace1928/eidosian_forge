from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('group_column_dtlike', [datetime.today(), datetime.today().date(), datetime.today().time()])
def test_apply_datetime_issue(group_column_dtlike, using_infer_string):
    df = DataFrame({'a': ['foo'], 'b': [group_column_dtlike]})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('a').apply(lambda x: Series(['spam'], index=[42]))
    dtype = 'string' if using_infer_string else 'object'
    expected = DataFrame(['spam'], Index(['foo'], dtype=dtype, name='a'), columns=[42])
    tm.assert_frame_equal(result, expected)