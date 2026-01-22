import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('series', [['1-1', '1-1', np.nan], ['1-1', '1-2', np.nan]])
def test_apply_categorical_with_nan_values(series, by_row):
    s = Series(series, dtype='category')
    if not by_row:
        msg = "'Series' object has no attribute 'split'"
        with pytest.raises(AttributeError, match=msg):
            s.apply(lambda x: x.split('-')[0], by_row=by_row)
        return
    result = s.apply(lambda x: x.split('-')[0], by_row=by_row)
    result = result.astype(object)
    expected = Series(['1', '1', np.nan], dtype='category')
    expected = expected.astype(object)
    tm.assert_series_equal(result, expected)