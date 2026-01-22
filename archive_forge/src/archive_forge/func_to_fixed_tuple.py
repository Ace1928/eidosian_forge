from numba.core import types, typing
from numba.core.cgutils import unpack_tuple
from numba.core.extending import intrinsic
from numba.core.imputils import impl_ret_new_ref
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
@intrinsic
def to_fixed_tuple(typingctx, array, length):
    """Convert *array* into a tuple of *length*

    Returns ``UniTuple(array.dtype, length)``

    ** Warning **
    - No boundchecking.
      If *length* is longer than *array.size*, the behavior is undefined.
    """
    if not isinstance(length, types.IntegerLiteral):
        raise RequireLiteralValue('*length* argument must be a constant')
    if array.ndim != 1:
        raise TypingError('Not supported on array.ndim={}'.format(array.ndim))
    tuple_size = int(length.literal_value)
    tuple_type = types.UniTuple(dtype=array.dtype, count=tuple_size)
    sig = tuple_type(array, length)

    def codegen(context, builder, signature, args):

        def impl(array, length, empty_tuple):
            out = empty_tuple
            for i in range(length):
                out = tuple_setitem(out, i, array[i])
            return out
        inner_argtypes = [signature.args[0], types.intp, tuple_type]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        ll_idx_type = context.get_value_type(types.intp)
        empty_tuple = context.get_constant_undef(tuple_type)
        inner_args = [args[0], ll_idx_type(tuple_size), empty_tuple]
        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res
    return (sig, codegen)