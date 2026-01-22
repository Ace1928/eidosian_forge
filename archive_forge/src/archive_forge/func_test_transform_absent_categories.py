import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', ['cumsum', 'cumprod', 'cummin', 'cummax'])
def test_transform_absent_categories(func):
    x_vals = [1]
    x_cats = range(2)
    y = [1]
    df = DataFrame({'x': Categorical(x_vals, x_cats), 'y': y})
    result = getattr(df.y.groupby(df.x, observed=False), func)()
    expected = df.y
    tm.assert_series_equal(result, expected)