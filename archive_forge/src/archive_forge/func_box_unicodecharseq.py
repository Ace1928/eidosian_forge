from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.UnicodeCharSeq)
def box_unicodecharseq(typ, val, c):
    unicode_kind = {1: c.pyapi.py_unicode_1byte_kind, 2: c.pyapi.py_unicode_2byte_kind, 4: c.pyapi.py_unicode_4byte_kind}[numpy_support.sizeof_unicode_char]
    kind = c.context.get_constant(types.int32, unicode_kind)
    rawptr = cgutils.alloca_once_value(c.builder, value=val)
    strptr = c.builder.bitcast(rawptr, c.pyapi.cstring)
    fullsize = c.context.get_constant(types.intp, typ.count)
    zero = fullsize.type(0)
    one = fullsize.type(1)
    step = fullsize.type(numpy_support.sizeof_unicode_char)
    count = cgutils.alloca_once_value(c.builder, zero)
    with cgutils.loop_nest(c.builder, [fullsize], fullsize.type) as [idx]:
        ch = c.builder.load(c.builder.gep(strptr, [c.builder.mul(idx, step)]))
        with c.builder.if_then(cgutils.is_not_null(c.builder, ch)):
            c.builder.store(c.builder.add(idx, one), count)
    strlen = c.builder.load(count)
    return c.pyapi.string_from_kind_and_data(kind, strptr, strlen)