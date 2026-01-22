import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_cython_agg_boolean():
    frame = DataFrame({'a': np.random.default_rng(2).integers(0, 5, 50), 'b': np.random.default_rng(2).integers(0, 2, 50).astype('bool')})
    result = frame.groupby('a')['b'].mean()
    msg = 'using SeriesGroupBy.mean'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = frame.groupby('a')['b'].agg(np.mean)
    tm.assert_series_equal(result, expected)