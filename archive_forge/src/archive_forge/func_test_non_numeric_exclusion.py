import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_non_numeric_exclusion(self, interp_method, request, using_array_manager):
    interpolation, method = interp_method
    df = DataFrame({'col1': ['A', 'A', 'B', 'B'], 'col2': [1, 2, 3, 4]})
    rs = df.quantile(0.5, numeric_only=True, interpolation=interpolation, method=method)
    xp = df.median(numeric_only=True).rename(0.5)
    if interpolation == 'nearest':
        xp = (xp + 0.5).astype(np.int64)
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    tm.assert_series_equal(rs, xp)