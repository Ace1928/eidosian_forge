import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_builtin(operator.is_, types.EnumMember, types.EnumMember)
def enum_is(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    if tu == tv:
        res = context.generic_compare(builder, operator.eq, (tu.dtype, tv.dtype), (u, v))
    else:
        res = context.get_constant(sig.return_type, False)
    return impl_ret_untracked(context, builder, sig.return_type, res)