import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('method', ['min', 'max'])
@pytest.mark.parametrize('box', [pd.Series, pd.array])
def test_min_max_numpy(method, box, dtype, request, arrow_string_storage):
    if dtype.storage in arrow_string_storage and box is pd.array:
        if box is pd.array:
            reason = "'<=' not supported between instances of 'str' and 'NoneType'"
        else:
            reason = "'ArrowStringArray' object has no attribute 'max'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.applymarker(mark)
    arr = box(['a', 'b', 'c', None], dtype=dtype)
    result = getattr(np, method)(arr)
    expected = 'a' if method == 'min' else 'c'
    assert result == expected