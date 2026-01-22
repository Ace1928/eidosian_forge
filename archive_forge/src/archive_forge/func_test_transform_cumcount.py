import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_cumcount():
    df = DataFrame({'a': [0, 0, 0, 1, 1, 1], 'b': range(6)})
    grp = df.groupby(np.repeat([0, 1], 3))
    result = grp.cumcount()
    expected = Series([0, 1, 2, 0, 1, 2])
    tm.assert_series_equal(result, expected)
    result = grp.transform('cumcount')
    tm.assert_series_equal(result, expected)