import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func, expected_values', [(Series.sort_values, [5, 4, 3, 2, 1]), (lambda x: x.head(1), [5.0, np.nan, 3, 2, np.nan])])
@pytest.mark.parametrize('keys', [['a1'], ['a1', 'a2']])
@pytest.mark.parametrize('keys_in_index', [True, False])
def test_transform_aligns(func, frame_or_series, expected_values, keys, keys_in_index):
    df = DataFrame({'a1': [1, 1, 3, 2, 2], 'b': [5, 4, 3, 2, 1]})
    if 'a2' in keys:
        df['a2'] = df['a1']
    if keys_in_index:
        df = df.set_index(keys, append=True)
    gb = df.groupby(keys)
    if frame_or_series is Series:
        gb = gb['b']
    result = gb.transform(func)
    expected = DataFrame({'b': expected_values}, index=df.index)
    if frame_or_series is Series:
        expected = expected['b']
    tm.assert_equal(result, expected)