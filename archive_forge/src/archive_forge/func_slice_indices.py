from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
@lower_builtin('slice.indices', types.SliceType, types.Integer)
def slice_indices(context, builder, sig, args):
    length = args[1]
    sli = context.make_helper(builder, sig.args[0], args[0])
    with builder.if_then(cgutils.is_neg_int(builder, length), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, ('length should not be negative',))
    with builder.if_then(cgutils.is_scalar_zero(builder, sli.step), likely=False):
        context.call_conv.return_user_exc(builder, ValueError, ('slice step cannot be zero',))
    fix_slice(builder, sli, length)
    return context.make_tuple(builder, sig.return_type, (sli.start, sli.stop, sli.step))