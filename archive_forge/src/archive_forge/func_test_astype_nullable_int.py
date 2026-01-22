import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_astype_nullable_int(dtype):
    arr = pd.array(['1', pd.NA, '3'], dtype=dtype)
    result = arr.astype('Int64')
    expected = pd.array([1, pd.NA, 3], dtype='Int64')
    tm.assert_extension_array_equal(result, expected)