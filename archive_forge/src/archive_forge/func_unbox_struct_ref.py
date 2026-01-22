from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
from numba.core.typing.templates import AttributeTemplate
@unbox(struct_type)
def unbox_struct_ref(typ, obj, c):
    mi_obj = c.pyapi.object_getattr_string(obj, '_meminfo')
    mip_type = types.MemInfoPointer(types.voidptr)
    mi = c.unbox(mip_type, mi_obj).value
    utils = _Utils(c.context, c.builder, typ)
    struct_ref = utils.new_struct_ref(mi)
    out = struct_ref._getvalue()
    c.pyapi.decref(mi_obj)
    return NativeValue(out)