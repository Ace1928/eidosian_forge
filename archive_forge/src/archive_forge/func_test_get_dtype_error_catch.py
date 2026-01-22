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
@pytest.mark.filterwarnings('ignore:is_categorical_dtype is deprecated:DeprecationWarning')
@pytest.mark.parametrize('func', get_is_dtype_funcs(), ids=lambda x: x.__name__)
def test_get_dtype_error_catch(func):
    msg = f'{func.__name__} is deprecated'
    warn = None
    if func is com.is_int64_dtype or func is com.is_interval_dtype or func is com.is_datetime64tz_dtype or (func is com.is_categorical_dtype) or (func is com.is_period_dtype):
        warn = DeprecationWarning
    with tm.assert_produces_warning(warn, match=msg):
        assert not func(None)