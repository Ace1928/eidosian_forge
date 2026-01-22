import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('keys', [['A1'], ['A1', 'A2']])
def test_null_group_lambda_self(sort, dropna, keys):
    size = 50
    nulls1 = np.random.default_rng(2).choice([False, True], size)
    nulls2 = np.random.default_rng(2).choice([False, True], size)
    nulls_grouper = nulls1 if len(keys) == 1 else nulls1 | nulls2
    a1 = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a1[nulls1] = np.nan
    a2 = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a2[nulls2] = np.nan
    values = np.random.default_rng(2).integers(0, 5, size=a1.shape)
    df = DataFrame({'A1': a1, 'A2': a2, 'B': values})
    expected_values = values
    if dropna and nulls_grouper.any():
        expected_values = expected_values.astype(float)
        expected_values[nulls_grouper] = np.nan
    expected = DataFrame(expected_values, columns=['B'])
    gb = df.groupby(keys, dropna=dropna, sort=sort)
    result = gb[['B']].transform(lambda x: x)
    tm.assert_frame_equal(result, expected)