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
def test_is_timedelta64_ns_dtype():
    assert not com.is_timedelta64_ns_dtype(np.dtype('m8[ps]'))
    assert not com.is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))
    assert com.is_timedelta64_ns_dtype(np.dtype('m8[ns]'))
    assert com.is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))