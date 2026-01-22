from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
@pytest.mark.parametrize('dtype', [str, float, np.int32, np.uint64, pd.Index([1, 2.0]), np.array(['a', 'b']), np.array([1, 2], dtype=np.uint32), 'int8', 'Int8', pd.Int8Dtype])
def test_is_not_int64_dtype(dtype):
    msg = 'is_int64_dtype is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not com.is_int64_dtype(dtype)