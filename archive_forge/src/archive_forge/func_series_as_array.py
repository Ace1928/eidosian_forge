import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@lower_builtin('__array__', SeriesType)
def series_as_array(context, builder, sig, args):
    val = make_series(context, builder, sig.args[0], ref=args[0])
    return val._get_ptr_by_name('values')