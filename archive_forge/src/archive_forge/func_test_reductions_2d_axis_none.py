import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
@pytest.mark.parametrize('method', ['mean', 'median', 'var', 'std', 'sum', 'prod'])
def test_reductions_2d_axis_none(self, data, method):
    arr2d = data.reshape(1, -1)
    err_expected = None
    err_result = None
    try:
        expected = getattr(data, method)()
    except Exception as err:
        err_expected = err
        try:
            result = getattr(arr2d, method)(axis=None)
        except Exception as err2:
            err_result = err2
    else:
        result = getattr(arr2d, method)(axis=None)
    if err_result is not None or err_expected is not None:
        assert type(err_result) == type(err_expected)
        return
    assert is_matching_na(result, expected) or result == expected