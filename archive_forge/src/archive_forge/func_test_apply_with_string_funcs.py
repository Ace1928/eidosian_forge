from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('func', ['sum', 'mean', 'min', 'max', 'std'])
@pytest.mark.parametrize('args,kwds', [pytest.param([], {}, id='no_args_or_kwds'), pytest.param([1], {}, id='axis_from_args'), pytest.param([], {'axis': 1}, id='axis_from_kwds'), pytest.param([], {'numeric_only': True}, id='optional_kwds'), pytest.param([1, True], {'numeric_only': True}, id='args_and_kwds')])
@pytest.mark.parametrize('how', ['agg', 'apply'])
def test_apply_with_string_funcs(request, float_frame, func, args, kwds, how):
    if len(args) > 1 and how == 'agg':
        request.applymarker(pytest.mark.xfail(raises=TypeError, reason='agg/apply signature mismatch - agg passes 2nd argument to func'))
    result = getattr(float_frame, how)(func, *args, **kwds)
    expected = getattr(float_frame, func)(*args, **kwds)
    tm.assert_series_equal(result, expected)