import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@lower_builtin('__array_wrap__', IndexType, types.Array)
def index_wrap_array(context, builder, sig, args):
    dest = make_index(context, builder, sig.return_type)
    dest.data = args[1]
    return impl_ret_borrowed(context, builder, sig.return_type, dest._getvalue())