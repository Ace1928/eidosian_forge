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
def objects_to_datetime64(data: np.ndarray, dayfirst, yearfirst, utc: bool=False, errors: DateTimeErrorChoices='raise', allow_object: bool=False, out_unit: str='ns'):
    """
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert/localize timestamps to UTC.
    errors : {'raise', 'ignore', 'coerce'}
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    out_unit : str, default "ns"

    Returns
    -------
    result : ndarray
        np.datetime64[out_unit] if returned values represent wall times or UTC
        timestamps.
        object if mixed timezones
    inferred_tz : tzinfo or None
        If not None, then the datetime64 values in `result` denote UTC timestamps.

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    TypeError  : When a type cannot be converted to datetime
    """
    assert errors in ['raise', 'ignore', 'coerce']
    data = np.array(data, copy=False, dtype=np.object_)
    result, tz_parsed = tslib.array_to_datetime(data, errors=errors, utc=utc, dayfirst=dayfirst, yearfirst=yearfirst, creso=abbrev_to_npy_unit(out_unit))
    if tz_parsed is not None:
        return (result, tz_parsed)
    elif result.dtype.kind == 'M':
        return (result, tz_parsed)
    elif result.dtype == object:
        if allow_object:
            return (result, tz_parsed)
        raise TypeError('DatetimeIndex has mixed timezones')
    else:
        raise TypeError(result)