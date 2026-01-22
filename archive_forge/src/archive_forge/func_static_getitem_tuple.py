import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin('static_getitem', types.LiteralStrKeyDict, types.StringLiteral)
@lower_builtin('static_getitem', types.LiteralList, types.IntegerLiteral)
@lower_builtin('static_getitem', types.LiteralList, types.SliceLiteral)
@lower_builtin('static_getitem', types.BaseTuple, types.IntegerLiteral)
@lower_builtin('static_getitem', types.BaseTuple, types.SliceLiteral)
def static_getitem_tuple(context, builder, sig, args):
    tupty, idxty = sig.args
    tup, idx = args
    if isinstance(idx, int):
        if idx < 0:
            idx += len(tupty)
        if not 0 <= idx < len(tupty):
            raise IndexError('cannot index at %d in %s' % (idx, tupty))
        res = builder.extract_value(tup, idx)
    elif isinstance(idx, slice):
        items = cgutils.unpack_tuple(builder, tup)[idx]
        res = context.make_tuple(builder, sig.return_type, items)
    elif isinstance(tupty, types.LiteralStrKeyDict):
        idx_val = idxty.literal_value
        idx_offset = tupty.fields.index(idx_val)
        res = builder.extract_value(tup, idx_offset)
    else:
        raise NotImplementedError('unexpected index %r for %s' % (idx, sig.args[0]))
    return impl_ret_borrowed(context, builder, sig.return_type, res)