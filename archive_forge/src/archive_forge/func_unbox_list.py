from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@unbox(types.List)
def unbox_list(typ, obj, c):
    """
    Convert list *obj* to a native list.

    If list was previously unboxed, we reuse the existing native list
    to ensure consistency.
    """
    size = c.pyapi.list_size(obj)
    errorptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    listptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))
    ptr = c.pyapi.object_get_private_data(obj)
    with c.builder.if_else(cgutils.is_not_null(c.builder, ptr)) as (has_meminfo, otherwise):
        with has_meminfo:
            list = listobj.ListInstance.from_meminfo(c.context, c.builder, typ, ptr)
            list.size = size
            if typ.reflected:
                list.parent = obj
            c.builder.store(list.value, listptr)
        with otherwise:
            _python_list_to_native(typ, obj, c, size, listptr, errorptr)

    def cleanup():
        c.pyapi.object_reset_private_data(obj)
    return NativeValue(c.builder.load(listptr), is_error=c.builder.load(errorptr), cleanup=cleanup)