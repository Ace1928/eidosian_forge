from numba.core import types, cgutils
from numba.core.imputils import (
@lower_builtin(enumerate, types.IterableType)
@lower_builtin(enumerate, types.IterableType, types.Integer)
def make_enumerate_object(context, builder, sig, args):
    assert len(args) == 1 or len(args) == 2
    srcty = sig.args[0]
    if len(args) == 1:
        src = args[0]
        start_val = context.get_constant(types.intp, 0)
    elif len(args) == 2:
        src = args[0]
        start_val = context.cast(builder, args[1], sig.args[1], types.intp)
    iterobj = call_getiter(context, builder, srcty, src)
    enum = context.make_helper(builder, sig.return_type)
    countptr = cgutils.alloca_once(builder, start_val.type)
    builder.store(start_val, countptr)
    enum.count = countptr
    enum.iter = iterobj
    res = enum._getvalue()
    return impl_ret_new_ref(context, builder, sig.return_type, res)