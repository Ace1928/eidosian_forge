import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_builtin(operator.ne, types.EnumMember, types.EnumMember)
def enum_ne(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    res = context.generic_compare(builder, operator.ne, (tu.dtype, tv.dtype), (u, v))
    return impl_ret_untracked(context, builder, sig.return_type, res)