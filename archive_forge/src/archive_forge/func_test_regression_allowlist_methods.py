from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
@pytest.mark.parametrize('op', AGG_FUNCTIONS)
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('sort', [True, False])
def test_regression_allowlist_methods(raw_frame, op, axis, skipna, sort):
    if axis == 0:
        frame = raw_frame
    else:
        frame = raw_frame.T
    if op in AGG_FUNCTIONS_WITH_SKIPNA:
        grouped = frame.groupby(level=0, axis=axis, sort=sort)
        result = getattr(grouped, op)(skipna=skipna)
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)(axis=axis, skipna=skipna))
        if sort:
            expected = expected.sort_index(axis=axis)
        tm.assert_frame_equal(result, expected)
    else:
        grouped = frame.groupby(level=0, axis=axis, sort=sort)
        result = getattr(grouped, op)()
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)(axis=axis))
        if sort:
            expected = expected.sort_index(axis=axis)
        tm.assert_frame_equal(result, expected)