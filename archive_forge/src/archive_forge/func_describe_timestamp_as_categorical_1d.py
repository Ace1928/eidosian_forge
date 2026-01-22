from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
def describe_timestamp_as_categorical_1d(data: Series, percentiles_ignored: Sequence[float]) -> Series:
    """Describe series containing timestamp data treated as categorical.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ['count', 'unique']
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    result: list[float | Timestamp] = [data.count(), count_unique]
    dtype = None
    if count_unique > 0:
        top, freq = (objcounts.index[0], objcounts.iloc[0])
        tz = data.dt.tz
        asint = data.dropna().values.view('i8')
        top = Timestamp(top)
        if top.tzinfo is not None and tz is not None:
            top = top.tz_convert(tz)
        else:
            top = top.tz_localize(tz)
        names += ['top', 'freq', 'first', 'last']
        result += [top, freq, Timestamp(asint.min(), tz=tz), Timestamp(asint.max(), tz=tz)]
    else:
        names += ['top', 'freq']
        result += [np.nan, np.nan]
        dtype = 'object'
    from pandas import Series
    return Series(result, index=names, name=data.name, dtype=dtype)