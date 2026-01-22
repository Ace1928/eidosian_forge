import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(operator.eq, types.BaseTuple, types.BaseTuple)
def tuple_eq(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    if len(tu.types) != len(tv.types):
        res = context.get_constant(types.boolean, False)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    res = context.get_constant(types.boolean, True)
    for i, (ta, tb) in enumerate(zip(tu.types, tv.types)):
        a = builder.extract_value(u, i)
        b = builder.extract_value(v, i)
        pred = context.generic_compare(builder, operator.eq, (ta, tb), (a, b))
        res = builder.and_(res, pred)
    return impl_ret_untracked(context, builder, sig.return_type, res)