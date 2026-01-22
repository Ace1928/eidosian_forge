import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_transform_with_nan_group():
    df = DataFrame({'a': range(10), 'b': [1, 1, 2, 3, np.nan, 4, 4, 5, 5, 5]})
    msg = 'using SeriesGroupBy.max'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(df.b)['a'].transform(max)
    expected = Series([1.0, 1.0, 2.0, 3.0, np.nan, 6.0, 6.0, 9.0, 9.0, 9.0], name='a')
    tm.assert_series_equal(result, expected)