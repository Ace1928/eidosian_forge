from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Integer)
def unbox_integer(typ, obj, c):
    ll_type = c.context.get_argument_type(typ)
    val = cgutils.alloca_once(c.builder, ll_type)
    longobj = c.pyapi.number_long(obj)
    with c.pyapi.if_object_ok(longobj):
        if typ.signed:
            llval = c.pyapi.long_as_longlong(longobj)
        else:
            llval = c.pyapi.long_as_ulonglong(longobj)
        c.pyapi.decref(longobj)
        c.builder.store(c.builder.trunc(llval, ll_type), val)
    return NativeValue(c.builder.load(val), is_error=c.pyapi.c_api_error())