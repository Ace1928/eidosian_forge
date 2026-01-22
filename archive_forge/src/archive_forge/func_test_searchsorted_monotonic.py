from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Period with BDay freq:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_searchsorted_monotonic(self, index_flat, request):
    index = index_flat
    if isinstance(index, pd.IntervalIndex):
        mark = pytest.mark.xfail(reason='IntervalIndex.searchsorted does not support Interval arg', raises=NotImplementedError)
        request.applymarker(mark)
    if index.empty:
        pytest.skip('Skip check for empty Index')
    value = index[0]
    expected_left, expected_right = (0, (index == value).argmin())
    if expected_right == 0:
        expected_right = len(index)
    if index.is_monotonic_increasing:
        ssm_left = index._searchsorted_monotonic(value, side='left')
        assert expected_left == ssm_left
        ssm_right = index._searchsorted_monotonic(value, side='right')
        assert expected_right == ssm_right
        ss_left = index.searchsorted(value, side='left')
        assert expected_left == ss_left
        ss_right = index.searchsorted(value, side='right')
        assert expected_right == ss_right
    elif index.is_monotonic_decreasing:
        ssm_left = index._searchsorted_monotonic(value, side='left')
        assert expected_left == ssm_left
        ssm_right = index._searchsorted_monotonic(value, side='right')
        assert expected_right == ssm_right
    else:
        msg = 'index must be monotonic increasing or decreasing'
        with pytest.raises(ValueError, match=msg):
            index._searchsorted_monotonic(value, side='left')