import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@typeof_impl.register(Series)
def typeof_series(val, c):
    index = typeof_impl(val._index, c)
    arrty = typeof_impl(val._values, c)
    assert arrty.ndim == 1
    assert arrty.layout == 'C'
    return SeriesType(arrty.dtype, index)