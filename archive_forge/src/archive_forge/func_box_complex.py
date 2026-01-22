from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Complex)
def box_complex(typ, val, c):
    cval = c.context.make_complex(c.builder, typ, value=val)
    if typ == types.complex64:
        freal = c.builder.fpext(cval.real, c.pyapi.double)
        fimag = c.builder.fpext(cval.imag, c.pyapi.double)
    else:
        assert typ == types.complex128
        freal, fimag = (cval.real, cval.imag)
    return c.pyapi.complex_from_doubles(freal, fimag)