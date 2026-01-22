from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Float)
def unbox_float(typ, obj, c):
    fobj = c.pyapi.number_float(obj)
    dbval = c.pyapi.float_as_double(fobj)
    c.pyapi.decref(fobj)
    if typ == types.float32:
        val = c.builder.fptrunc(dbval, c.context.get_argument_type(typ))
    else:
        assert typ == types.float64
        val = dbval
    return NativeValue(val, is_error=c.pyapi.c_api_error())