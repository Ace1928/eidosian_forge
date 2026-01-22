from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('klass', [list, tuple, np.array])
def test_where_array_like(self, klass):
    df = DataFrame({'a': [1, 2, 3]})
    cond = [[False], [True], [True]]
    expected = DataFrame({'a': [np.nan, 2, 3]})
    result = df.where(klass(cond))
    tm.assert_frame_equal(result, expected)
    df['b'] = 2
    expected['b'] = [2, np.nan, 2]
    cond = [[False, True], [True, False], [True, True]]
    result = df.where(klass(cond))
    tm.assert_frame_equal(result, expected)