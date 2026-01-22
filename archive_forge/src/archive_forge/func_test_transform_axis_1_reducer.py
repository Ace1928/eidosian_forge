import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_axis_1_reducer(request, reduction_func):
    if reduction_func in ('corrwith', 'ngroup', 'nth'):
        marker = pytest.mark.xfail(reason='transform incorrectly fails - GH#45986')
        request.applymarker(marker)
    df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}, index=['x', 'y'])
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby([0, 0, 1], axis=1)
    result = gb.transform(reduction_func)
    expected = df.T.groupby([0, 0, 1]).transform(reduction_func).T
    tm.assert_equal(result, expected)