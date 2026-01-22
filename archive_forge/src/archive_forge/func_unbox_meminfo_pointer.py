from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.MemInfoPointer)
def unbox_meminfo_pointer(typ, obj, c):
    res = c.pyapi.nrt_meminfo_from_pyobject(obj)
    errored = cgutils.is_null(c.builder, res)
    return NativeValue(res, is_error=errored)