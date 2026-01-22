import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_cast(types.BaseTuple, types.BaseTuple)
def tuple_to_tuple(context, builder, fromty, toty, val):
    if isinstance(fromty, types.BaseNamedTuple) or isinstance(toty, types.BaseNamedTuple):
        raise NotImplementedError
    if len(fromty) != len(toty):
        raise NotImplementedError
    olditems = cgutils.unpack_tuple(builder, val, len(fromty))
    items = [context.cast(builder, v, f, t) for v, f, t in zip(olditems, fromty, toty)]
    return context.make_tuple(builder, toty, items)