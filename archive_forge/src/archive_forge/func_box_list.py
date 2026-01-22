from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.List)
def box_list(typ, val, c):
    """
    Convert native list *val* to a list object.
    """
    list = listobj.ListInstance(c.context, c.builder, typ, val)
    obj = list.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    with c.builder.if_else(cgutils.is_not_null(c.builder, obj)) as (has_parent, otherwise):
        with has_parent:
            c.pyapi.incref(obj)
        with otherwise:
            nitems = list.size
            obj = c.pyapi.list_new(nitems)
            with c.builder.if_then(cgutils.is_not_null(c.builder, obj), likely=True):
                with cgutils.for_range(c.builder, nitems) as loop:
                    item = list.getitem(loop.index)
                    list.incref_value(item)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_setitem(obj, loop.index, itemobj)
            c.builder.store(obj, res)
    c.context.nrt.decref(c.builder, typ, val)
    return c.builder.load(res)