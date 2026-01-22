import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm

    Check a group transform that executes a cumulative function.

    Parameters
    ----------
    pd_op : callable
        The pandas cumulative function.
    np_op : callable
        The analogous one in NumPy.
    dtype : type
        The specified dtype of the data.
    