import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', [np.any, np.all])
def test_any_all_np_func(func):
    df = DataFrame([['foo', True], [np.nan, True], ['foo', True]], columns=['key', 'val'])
    exp = Series([True, np.nan, True], name='val')
    msg = 'using SeriesGroupBy.[any|all]'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.groupby('key')['val'].transform(func)
    tm.assert_series_equal(res, exp)