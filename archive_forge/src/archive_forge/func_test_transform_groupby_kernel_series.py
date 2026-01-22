from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('op', series_transform_kernels)
def test_transform_groupby_kernel_series(request, string_series, op):
    if op == 'ngroup':
        request.applymarker(pytest.mark.xfail(raises=ValueError, reason='ngroup not valid for NDFrame'))
    args = [0.0] if op == 'fillna' else []
    ones = np.ones(string_series.shape[0])
    warn = FutureWarning if op == 'fillna' else None
    msg = 'SeriesGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        expected = string_series.groupby(ones).transform(op, *args)
    result = string_series.transform(op, 0, *args)
    tm.assert_series_equal(result, expected)