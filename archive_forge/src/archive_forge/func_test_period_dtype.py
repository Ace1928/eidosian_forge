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
@pytest.mark.parametrize('dtype', ['period[D]', 'period[3M]', 'period[us]', 'Period[D]', 'Period[3M]', 'Period[us]'])
def test_period_dtype(self, dtype):
    assert com.pandas_dtype(dtype) is not PeriodDtype(dtype)
    assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
    assert com.pandas_dtype(dtype) == dtype