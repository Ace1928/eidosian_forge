from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@reflect(types.List)
def reflect_list(typ, val, c):
    """
    Reflect the native list's contents into the Python object.
    """
    if not typ.reflected:
        return
    if typ.dtype.reflected:
        msg = 'cannot reflect element of reflected container: {}\n'.format(typ)
        raise TypeError(msg)
    list = listobj.ListInstance(c.context, c.builder, typ, val)
    with c.builder.if_then(list.dirty, likely=False):
        obj = list.parent
        size = c.pyapi.list_size(obj)
        new_size = list.size
        diff = c.builder.sub(new_size, size)
        diff_gt_0 = c.builder.icmp_signed('>=', diff, ir.Constant(diff.type, 0))
        with c.builder.if_else(diff_gt_0) as (if_grow, if_shrink):
            with if_grow:
                with cgutils.for_range(c.builder, size) as loop:
                    item = list.getitem(loop.index)
                    list.incref_value(item)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_setitem(obj, loop.index, itemobj)
                with cgutils.for_range(c.builder, diff) as loop:
                    idx = c.builder.add(size, loop.index)
                    item = list.getitem(idx)
                    list.incref_value(item)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_append(obj, itemobj)
                    c.pyapi.decref(itemobj)
            with if_shrink:
                c.pyapi.list_setslice(obj, new_size, size, None)
                with cgutils.for_range(c.builder, new_size) as loop:
                    item = list.getitem(loop.index)
                    list.incref_value(item)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_setitem(obj, loop.index, itemobj)
        list.set_dirty(False)