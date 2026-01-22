import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('agg', ['min', 'max', 'count', 'sum', 'prod', 'var', 'mean', 'median', 'ohlc', 'cumprod', 'cumsum', 'shift', 'any', 'all', 'quantile', 'first', 'last', 'rank', 'cummin', 'cummax'])
def test_read_only_buffer_source_agg(agg):
    df = DataFrame({'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0], 'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']})
    df._mgr.arrays[0].flags.writeable = False
    result = df.groupby(['species']).agg({'sepal_length': agg})
    expected = df.copy().groupby(['species']).agg({'sepal_length': agg})
    tm.assert_equal(result, expected)