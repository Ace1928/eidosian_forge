import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.filterwarnings('ignore:RuntimeWarning')
@pytest.mark.parametrize('f', [lambda x, y: x.expanding().cov(y, pairwise=False), lambda x, y: x.expanding().corr(y, pairwise=False), lambda x, y: x.rolling(window=3).cov(y, pairwise=False), lambda x, y: x.rolling(window=3).corr(y, pairwise=False), lambda x, y: x.ewm(com=3).cov(y, pairwise=False), lambda x, y: x.ewm(com=3).corr(y, pairwise=False)])
def test_no_pairwise_with_other(self, pairwise_frames, pairwise_other_frame, f):
    result = f(pairwise_frames, pairwise_other_frame) if pairwise_frames.columns.is_unique else None
    if result is not None:
        expected_index = pairwise_frames.index.union(pairwise_other_frame.index)
        expected_columns = pairwise_frames.columns.union(pairwise_other_frame.columns)
        tm.assert_index_equal(result.index, expected_index)
        tm.assert_index_equal(result.columns, expected_columns)
    else:
        with pytest.raises(ValueError, match="'arg1' columns are not unique"):
            f(pairwise_frames, pairwise_other_frame)
        with pytest.raises(ValueError, match="'arg2' columns are not unique"):
            f(pairwise_other_frame, pairwise_frames)