from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('op', ['mean', 'median', 'std', 'var'])
@pytest.mark.parametrize('how', ['agg', 'apply'])
def test_apply_np_reducer(op, how):
    float_frame = DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = getattr(float_frame, how)(op)
    kwargs = {'ddof': 1} if op in ('std', 'var') else {}
    expected = Series(getattr(np, op)(float_frame, axis=0, **kwargs), index=float_frame.columns)
    tm.assert_series_equal(result, expected)