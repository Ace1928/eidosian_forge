import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_constant(types.Tuple)
@lower_constant(types.NamedTuple)
def unituple_constant(context, builder, ty, pyval):
    """
    Create a heterogeneous tuple constant.
    """
    consts = [context.get_constant_generic(builder, ty.types[i], v) for i, v in enumerate(pyval)]
    return impl_ret_borrowed(context, builder, ty, cgutils.pack_struct(builder, consts))