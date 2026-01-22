import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@overload_method(SeriesType, 'clip')
def series_clip(series, lower, upper):
    """
    Series.clip(...)
    """

    def clip_impl(series, lower, upper):
        data = series._values.copy()
        for i in range(len(data)):
            v = data[i]
            if v < lower:
                data[i] = lower
            elif v > upper:
                data[i] = upper
        return Series(data, series._index)
    return clip_impl