import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(operator.add, types.BaseTuple, types.BaseTuple)
def tuple_add(context, builder, sig, args):
    left, right = [cgutils.unpack_tuple(builder, x) for x in args]
    res = context.make_tuple(builder, sig.return_type, left + right)
    return impl_ret_borrowed(context, builder, sig.return_type, res)