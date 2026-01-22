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
@pytest.mark.parametrize('dtype', [str, float, np.datetime64, np.timedelta64, pd.Index([1, 2.0]), np.array(['a', 'b']), np.array([], dtype=np.timedelta64)] + tm.UNSIGNED_INT_NUMPY_DTYPES + to_numpy_dtypes(tm.UNSIGNED_INT_NUMPY_DTYPES) + tm.UNSIGNED_INT_EA_DTYPES + to_ea_dtypes(tm.UNSIGNED_INT_EA_DTYPES))
def test_is_not_signed_integer_dtype(dtype):
    assert not com.is_signed_integer_dtype(dtype)