import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.xfail(reason='Not implemented StringArray.sum')
def test_reduce_missing(skipna, dtype):
    arr = pd.Series([None, 'a', None, 'b', 'c', None], dtype=dtype)
    result = arr.sum(skipna=skipna)
    if skipna:
        assert result == 'abc'
    else:
        assert pd.isna(result)