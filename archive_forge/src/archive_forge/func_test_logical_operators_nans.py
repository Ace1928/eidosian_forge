import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('left, right, op, expected', [([True, False, np.nan], [True, False, True], operator.and_, [True, False, False]), ([True, False, True], [True, False, np.nan], operator.and_, [True, False, False]), ([True, False, np.nan], [True, False, True], operator.or_, [True, False, False]), ([True, False, True], [True, False, np.nan], operator.or_, [True, False, True])])
def test_logical_operators_nans(self, left, right, op, expected, frame_or_series):
    result = op(frame_or_series(left), frame_or_series(right))
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)