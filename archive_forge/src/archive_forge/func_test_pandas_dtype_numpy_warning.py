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
def test_pandas_dtype_numpy_warning():
    with tm.assert_produces_warning(DeprecationWarning, check_stacklevel=False, match='Converting `np.integer` or `np.signedinteger` to a dtype is deprecated'):
        pandas_dtype(np.integer)