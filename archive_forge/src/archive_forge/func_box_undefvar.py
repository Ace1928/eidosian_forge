from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.UndefVar)
def box_undefvar(typ, val, c):
    """This type cannot be boxed, there's no Python equivalent"""
    msg = 'UndefVar type cannot be boxed, there is no Python equivalent of this type.'
    raise TypingError(msg)