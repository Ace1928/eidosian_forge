from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', ['first', 'last'])
def test_groupby_last_first_preserve_categoricaldtype(func):
    df = DataFrame({'a': [1, 2, 3]})
    df['b'] = df['a'].astype('category')
    result = getattr(df.groupby('a')['b'], func)()
    expected = Series(Categorical([1, 2, 3]), name='b', index=Index([1, 2, 3], name='a'))
    tm.assert_series_equal(expected, result)