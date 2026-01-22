from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('arr, correct', [('arr_complex', False), ('arr_int', False), ('arr_bool', False), ('arr_str', False), ('arr_utf', False), ('arr_complex', False), ('arr_complex_nan', False), ('arr_nan_nanj', False), ('arr_nan_infj', True), ('arr_complex_nan_infj', True)])
def test_has_infs_non_float(request, arr, correct, disable_bottleneck):
    val = request.getfixturevalue(arr)
    while getattr(val, 'ndim', True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0
        if not hasattr(val, 'ndim'):
            break
        val = np.take(val, 0, axis=-1)