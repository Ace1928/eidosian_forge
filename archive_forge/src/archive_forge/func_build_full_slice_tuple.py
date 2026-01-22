from numba.core import types, typing, errors
from numba.core.cgutils import alloca_once
from numba.core.extending import intrinsic
@intrinsic
def build_full_slice_tuple(tyctx, sz):
    """Creates a sz-tuple of full slices."""
    if not isinstance(sz, types.IntegerLiteral):
        raise errors.RequireLiteralValue(sz)
    size = int(sz.literal_value)
    tuple_type = types.UniTuple(dtype=types.slice2_type, count=size)
    sig = tuple_type(sz)

    def codegen(context, builder, signature, args):

        def impl(length, empty_tuple):
            out = empty_tuple
            for i in range(length):
                out = tuple_setitem(out, i, slice(None, None))
            return out
        inner_argtypes = [types.intp, tuple_type]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        ll_idx_type = context.get_value_type(types.intp)
        empty_tuple = context.get_constant_undef(tuple_type)
        inner_args = [ll_idx_type(size), empty_tuple]
        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res
    return (sig, codegen)