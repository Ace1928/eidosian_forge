import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('f', [lambda x: x.cov(), lambda x: x.corr()])
def test_no_flex(self, pairwise_frames, pairwise_target_frame, f):
    result = f(pairwise_frames)
    tm.assert_index_equal(result.index, pairwise_frames.columns)
    tm.assert_index_equal(result.columns, pairwise_frames.columns)
    expected = f(pairwise_target_frame)
    result = result.dropna().values
    expected = expected.dropna().values
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)