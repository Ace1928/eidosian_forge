from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
def maybe_convert_dtype(data, copy: bool, tz: tzinfo | None=None):
    """
    Convert data based on dtype conventions, issuing
    errors where appropriate.

    Parameters
    ----------
    data : np.ndarray or pd.Index
    copy : bool
    tz : tzinfo or None, default None

    Returns
    -------
    data : np.ndarray or pd.Index
    copy : bool

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    if not hasattr(data, 'dtype'):
        return (data, copy)
    if is_float_dtype(data.dtype):
        data = data.astype(DT64NS_DTYPE).view('i8')
        copy = False
    elif lib.is_np_dtype(data.dtype, 'm') or is_bool_dtype(data.dtype):
        raise TypeError(f'dtype {data.dtype} cannot be converted to datetime64[ns]')
    elif isinstance(data.dtype, PeriodDtype):
        raise TypeError('Passing PeriodDtype data is invalid. Use `data.to_timestamp()` instead')
    elif isinstance(data.dtype, ExtensionDtype) and (not isinstance(data.dtype, DatetimeTZDtype)):
        data = np.array(data, dtype=np.object_)
        copy = False
    return (data, copy)