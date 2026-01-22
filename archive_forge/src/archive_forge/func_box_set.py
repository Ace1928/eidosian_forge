from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Set)
def box_set(typ, val, c):
    """
    Convert native set *val* to a set object.
    """
    inst = setobj.SetInstance(c.context, c.builder, typ, val)
    obj = inst.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    with c.builder.if_else(cgutils.is_not_null(c.builder, obj)) as (has_parent, otherwise):
        with has_parent:
            c.pyapi.incref(obj)
        with otherwise:
            payload = inst.payload
            ok, listobj = _native_set_to_python_list(typ, payload, c)
            with c.builder.if_then(ok, likely=True):
                obj = c.pyapi.set_new(listobj)
                c.pyapi.decref(listobj)
                c.builder.store(obj, res)
    c.context.nrt.decref(c.builder, typ, val)
    return c.builder.load(res)