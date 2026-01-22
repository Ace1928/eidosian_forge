from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.DeferredType)
def box_deferred(typ, val, c):
    out = c.pyapi.from_native_value(typ.get(), c.builder.extract_value(val, [0]), env_manager=c.env_manager)
    return out