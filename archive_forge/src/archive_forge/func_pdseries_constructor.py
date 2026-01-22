import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@lower_builtin(Series, types.Array, IndexType)
def pdseries_constructor(context, builder, sig, args):
    data, index = args
    series = make_series(context, builder, sig.return_type)
    series.index = index
    series.values = data
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())