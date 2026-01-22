import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
def get_reduction_result_dtype(dtype):
    if dtype.itemsize == 8:
        return dtype
    elif dtype.kind in 'ib':
        return NUMPY_INT_TO_DTYPE[np.dtype(int)]
    else:
        return NUMPY_INT_TO_DTYPE[np.dtype('uint')]