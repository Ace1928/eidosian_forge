from numba.core import types, cgutils
from numba.core.imputils import (
@lower_builtin('iternext', types.Generator)
@iternext_impl(RefType.BORROWED)
def iternext_zip(context, builder, sig, args, result):
    genty, = sig.args
    gen, = args
    impl = context.get_generator_impl(genty)
    status, retval = impl(context, builder, sig, args)
    context.add_linking_libs(getattr(impl, 'libs', ()))
    with cgutils.if_likely(builder, status.is_ok):
        result.set_valid(True)
        result.yield_(retval)
    with cgutils.if_unlikely(builder, status.is_stop_iteration):
        result.set_exhausted()
    with cgutils.if_unlikely(builder, builder.and_(status.is_error, builder.not_(status.is_stop_iteration))):
        context.call_conv.return_status_propagate(builder, status)