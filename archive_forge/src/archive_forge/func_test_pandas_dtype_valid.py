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
@pytest.mark.parametrize('dtype', [object, 'float64', np.object_, np.dtype('object'), 'O', np.float64, float, np.dtype('float64'), 'object_'])
def test_pandas_dtype_valid(self, dtype):
    assert com.pandas_dtype(dtype) == dtype