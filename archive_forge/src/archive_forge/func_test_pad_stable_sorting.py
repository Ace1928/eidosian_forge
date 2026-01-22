import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('fill_method', ['ffill', 'bfill'])
def test_pad_stable_sorting(fill_method):
    x = [0] * 20
    y = [np.nan] * 10 + [1] * 10
    if fill_method == 'bfill':
        y = y[::-1]
    df = DataFrame({'x': x, 'y': y})
    expected = df.drop('x', axis=1)
    result = getattr(df.groupby('x'), fill_method)()
    tm.assert_frame_equal(result, expected)