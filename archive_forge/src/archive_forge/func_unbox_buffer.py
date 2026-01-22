from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Buffer)
def unbox_buffer(typ, obj, c):
    """
    Convert a Py_buffer-providing object to a native array structure.
    """
    buf = c.pyapi.alloca_buffer()
    res = c.pyapi.get_buffer(obj, buf)
    is_error = cgutils.is_not_null(c.builder, res)
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()
    with cgutils.if_likely(c.builder, c.builder.not_(is_error)):
        ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
        if c.context.enable_nrt:
            c.pyapi.nrt_adapt_buffer_from_python(buf, ptr)
        else:
            c.pyapi.numba_buffer_adaptor(buf, ptr)

    def cleanup():
        c.pyapi.release_buffer(buf)
    return NativeValue(c.builder.load(aryptr), is_error=is_error, cleanup=cleanup)