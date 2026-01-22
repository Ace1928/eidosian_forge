from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.Boolean)
def unbox_boolean(typ, obj, c):
    istrue = c.pyapi.object_istrue(obj)
    zero = ir.Constant(istrue.type, 0)
    val = c.builder.icmp_signed('!=', istrue, zero)
    return NativeValue(val, is_error=c.pyapi.c_api_error())