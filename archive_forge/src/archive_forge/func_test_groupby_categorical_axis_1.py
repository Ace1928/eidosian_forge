from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('code', [[1, 0, 0], [0, 0, 0]])
def test_groupby_categorical_axis_1(code):
    df = DataFrame({'a': [1, 2, 3, 4], 'b': [-1, -2, -3, -4], 'c': [5, 6, 7, 8]})
    cat = Categorical.from_codes(code, categories=list('abc'))
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(cat, axis=1, observed=False)
    result = gb.mean()
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb2 = df.T.groupby(cat, axis=0, observed=False)
    expected = gb2.mean().T
    tm.assert_frame_equal(result, expected)