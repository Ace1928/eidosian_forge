import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_multi(self, interp_method, request, using_array_manager):
    interpolation, method = interp_method
    df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=['a', 'b', 'c'])
    result = df.quantile([0.25, 0.5], interpolation=interpolation, method=method)
    expected = DataFrame([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]], index=[0.25, 0.5], columns=['a', 'b', 'c'])
    if interpolation == 'nearest':
        expected = expected.astype(np.int64)
    if method == 'table' and using_array_manager:
        request.applymarker(pytest.mark.xfail(reason='Axis name incorrectly set.'))
    tm.assert_frame_equal(result, expected)