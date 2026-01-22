import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('ops, names', [([np.sum], ['sum']), ([np.sum, np.mean], ['sum', 'mean']), (np.array([np.sum]), ['sum']), (np.array([np.sum, np.mean]), ['sum', 'mean'])])
@pytest.mark.parametrize('how, kwargs', [['agg', {}], ['apply', {'by_row': 'compat'}], ['apply', {'by_row': False}]])
def test_apply_listlike_reducer(string_series, ops, names, how, kwargs):
    expected = Series({name: op(string_series) for name, op in zip(names, ops)})
    expected.name = 'series'
    warn = FutureWarning if how == 'agg' else None
    msg = f'using Series.[{'|'.join(names)}]'
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)