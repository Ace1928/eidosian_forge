from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('df, func, expected', chain(tm.get_cython_table_params(DataFrame(), [('cumprod', DataFrame()), ('cumsum', DataFrame())]), tm.get_cython_table_params(DataFrame([[np.nan, 1], [1, 2]]), [('cumprod', DataFrame([[np.nan, 1], [1, 2]])), ('cumsum', DataFrame([[np.nan, 1], [1, 3]]))])))
def test_agg_cython_table_transform_frame(df, func, expected, axis):
    if axis in ('columns', 1):
        expected = expected.astype('float64')
    warn = None if isinstance(func, str) else FutureWarning
    with tm.assert_produces_warning(warn, match='is currently using DataFrame.*'):
        result = df.agg(func, axis=axis)
    tm.assert_frame_equal(result, expected)