from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('how', ['inner', 'left', 'outer', 'right'])
@pytest.mark.parametrize('tz', [None, 'America/Chicago'])
def test_merge_datetime_different_resolution(tz, how):
    vals = [pd.Timestamp(2023, 5, 12, tz=tz), pd.Timestamp(2023, 5, 13, tz=tz), pd.Timestamp(2023, 5, 14, tz=tz)]
    df1 = DataFrame({'t': vals[:2], 'a': [1.0, 2.0]})
    df1['t'] = df1['t'].dt.as_unit('ns')
    df2 = DataFrame({'t': vals[1:], 'b': [1.0, 2.0]})
    df2['t'] = df2['t'].dt.as_unit('s')
    expected = DataFrame({'t': vals, 'a': [1.0, 2.0, np.nan], 'b': [np.nan, 1.0, 2.0]})
    expected['t'] = expected['t'].dt.as_unit('ns')
    if how == 'inner':
        expected = expected.iloc[[1]].reset_index(drop=True)
    elif how == 'left':
        expected = expected.iloc[[0, 1]]
    elif how == 'right':
        expected = expected.iloc[[1, 2]].reset_index(drop=True)
    result = df1.merge(df2, on='t', how=how)
    tm.assert_frame_equal(result, expected)