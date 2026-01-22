import collections
from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
@pytest.mark.parametrize('null_obj', [np.nan, None])
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_value_counts_null(null_obj, index_or_series_obj):
    orig = index_or_series_obj
    obj = orig.copy()
    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(orig, MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")
    values = obj._values
    values[0:2] = null_obj
    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)
    counter = collections.Counter(obj.dropna())
    expected = Series(dict(counter.most_common()), dtype=np.int64, name='count')
    if obj.dtype != np.float16:
        expected.index = expected.index.astype(obj.dtype)
    else:
        with pytest.raises(NotImplementedError, match='float16 indexes are not '):
            expected.index.astype(obj.dtype)
        return
    expected.index.name = obj.name
    result = obj.value_counts()
    if obj.duplicated().any():
        expected = expected.sort_index()
        result = result.sort_index()
    if not isinstance(result.dtype, np.dtype):
        if getattr(obj.dtype, 'storage', '') == 'pyarrow':
            expected = expected.astype('int64[pyarrow]')
        else:
            expected = expected.astype('Int64')
    tm.assert_series_equal(result, expected)
    expected[null_obj] = 3
    result = obj.value_counts(dropna=False)
    if obj.duplicated().any():
        expected = expected.sort_index()
        result = result.sort_index()
    tm.assert_series_equal(result, expected)