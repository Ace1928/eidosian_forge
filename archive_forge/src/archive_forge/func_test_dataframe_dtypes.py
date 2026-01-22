import numpy as np
import pytest
from pandas.errors import DataError
from pandas.core.dtypes.common import pandas_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method, expected_data, min_periods', [('count', {0: Series([1, 2, 2, 2, 2]), 1: Series([1, 2, 2, 2, 2])}, 0), ('max', {0: Series([np.nan, 2, 4, 6, 8]), 1: Series([np.nan, 3, 5, 7, 9])}, None), ('min', {0: Series([np.nan, 0, 2, 4, 6]), 1: Series([np.nan, 1, 3, 5, 7])}, None), ('sum', {0: Series([np.nan, 2, 6, 10, 14]), 1: Series([np.nan, 4, 8, 12, 16])}, None), ('mean', {0: Series([np.nan, 1, 3, 5, 7]), 1: Series([np.nan, 2, 4, 6, 8])}, None), ('std', {0: Series([np.nan] + [np.sqrt(2)] * 4), 1: Series([np.nan] + [np.sqrt(2)] * 4)}, None), ('var', {0: Series([np.nan, 2, 2, 2, 2]), 1: Series([np.nan, 2, 2, 2, 2])}, None), ('median', {0: Series([np.nan, 1, 3, 5, 7]), 1: Series([np.nan, 2, 4, 6, 8])}, None)])
def test_dataframe_dtypes(method, expected_data, dtypes, min_periods, step):
    df = DataFrame(np.arange(10).reshape((5, 2)), dtype=get_dtype(dtypes))
    rolled = df.rolling(2, min_periods=min_periods, step=step)
    if dtypes in ('m8[ns]', 'M8[ns]', 'datetime64[ns, UTC]') and method != 'count':
        msg = 'Cannot aggregate non-numeric type'
        with pytest.raises(DataError, match=msg):
            getattr(rolled, method)()
    else:
        result = getattr(rolled, method)()
        expected = DataFrame(expected_data, dtype='float64')[::step]
        tm.assert_frame_equal(result, expected)