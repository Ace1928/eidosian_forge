from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def isna_all(arr: ArrayLike) -> bool:
    """
    Optimized equivalent to isna(arr).all()
    """
    total_len = len(arr)
    chunk_len = max(total_len // 40, 1000)
    dtype = arr.dtype
    if lib.is_np_dtype(dtype, 'f'):
        checker = nan_checker
    elif lib.is_np_dtype(dtype, 'mM') or isinstance(dtype, (DatetimeTZDtype, PeriodDtype)):
        checker = lambda x: np.asarray(x.view('i8')) == iNaT
    else:
        checker = lambda x: _isna_array(x, inf_as_na=INF_AS_NA)
    return all((checker(arr[i:i + chunk_len]).all() for i in range(0, total_len, chunk_len)))