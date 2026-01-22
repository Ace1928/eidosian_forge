import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
@pytest.mark.parametrize('np_dtype', ['int64', 'uint64', 'float32', 'float64'])
def test_cython_group_transform_cumsum(np_dtype):
    dtype = np.dtype(np_dtype).type
    pd_op, np_op = (group_cumsum, np.cumsum)
    _check_cython_group_transform_cumulative(pd_op, np_op, dtype)