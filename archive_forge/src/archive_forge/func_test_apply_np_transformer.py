from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('op', ['abs', 'ceil', 'cos', 'cumsum', 'exp', 'log', 'sqrt', 'square'])
@pytest.mark.parametrize('how', ['transform', 'apply'])
def test_apply_np_transformer(float_frame, op, how):
    float_frame.iloc[0, 0] = -1.0
    warn = None
    if op in ['log', 'sqrt']:
        warn = RuntimeWarning
    with tm.assert_produces_warning(warn, check_stacklevel=False):
        result = getattr(float_frame, how)(op)
        expected = getattr(np, op)(float_frame)
    tm.assert_frame_equal(result, expected)