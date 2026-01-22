import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interpolate_inplace(self, frame_or_series, using_array_manager, request):
    if using_array_manager and frame_or_series is DataFrame:
        mark = pytest.mark.xfail(reason='.values-based in-place check is invalid')
        request.applymarker(mark)
    obj = frame_or_series([1, np.nan, 2])
    orig = obj.values
    obj.interpolate(inplace=True)
    expected = frame_or_series([1, 1.5, 2])
    tm.assert_equal(obj, expected)
    assert np.shares_memory(orig, obj.values)
    assert orig.squeeze()[1] == 1.5