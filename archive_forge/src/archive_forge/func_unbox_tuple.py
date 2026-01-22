from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.BaseTuple)
def unbox_tuple(typ, obj, c):
    """
    Convert tuple *obj* to a native array (if homogeneous) or structure.
    """
    n = len(typ)
    values = []
    cleanups = []
    lty = c.context.get_value_type(typ)
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    value_ptr = cgutils.alloca_once(c.builder, lty)
    actual_size = c.pyapi.tuple_size(obj)
    size_matches = c.builder.icmp_unsigned('==', actual_size, ir.Constant(actual_size.type, n))
    with c.builder.if_then(c.builder.not_(size_matches), likely=False):
        c.pyapi.err_format('PyExc_ValueError', 'size mismatch for tuple, expected %d element(s) but got %%zd' % (n,), actual_size)
        c.builder.store(cgutils.true_bit, is_error_ptr)
    for i, eltype in enumerate(typ):
        elem = c.pyapi.tuple_getitem(obj, i)
        native = c.unbox(eltype, elem)
        values.append(native.value)
        with c.builder.if_then(native.is_error, likely=False):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        if native.cleanup is not None:
            cleanups.append(native.cleanup)
    value = c.context.make_tuple(c.builder, typ, values)
    c.builder.store(value, value_ptr)
    if cleanups:
        with c.builder.if_then(size_matches, likely=True):

            def cleanup():
                for func in reversed(cleanups):
                    func()
    else:
        cleanup = None
    return NativeValue(c.builder.load(value_ptr), cleanup=cleanup, is_error=c.builder.load(is_error_ptr))