import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_length():
    df = DataFrame({'col1': [1, 1, 2, 2], 'col2': [1, 2, 3, np.nan]})
    expected = Series([3.0] * 4)

    def nsum(x):
        return np.nansum(x)
    msg = 'using DataFrameGroupBy.sum'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        results = [df.groupby('col1').transform(sum)['col2'], df.groupby('col1')['col2'].transform(sum), df.groupby('col1').transform(nsum)['col2'], df.groupby('col1')['col2'].transform(nsum)]
    for result in results:
        tm.assert_series_equal(result, expected, check_names=False)