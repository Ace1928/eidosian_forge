import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_cython_group_transform_cumprod():
    dtype = np.float64
    pd_op, np_op = (group_cumprod, np.cumprod)
    _check_cython_group_transform_cumulative(pd_op, np_op, dtype)