from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
def test_infer_dtype_from_datetime():
    dt64 = np.datetime64(1, 'ns')
    dtype, val = infer_dtype_from_scalar(dt64)
    assert dtype == 'M8[ns]'
    ts = Timestamp(1)
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == 'M8[ns]'
    dt = datetime(2000, 1, 1, 0, 0)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == 'M8[us]'