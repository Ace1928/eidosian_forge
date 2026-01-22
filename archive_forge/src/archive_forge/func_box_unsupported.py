from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def box_unsupported(typ, val, c):
    msg = 'cannot convert native %s to Python object' % (typ,)
    c.pyapi.err_set_string('PyExc_TypeError', msg)
    res = c.pyapi.get_null_object()
    return res