import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_builtin('static_getitem', types.EnumClass, types.StringLiteral)
def enum_class_getitem(context, builder, sig, args):
    """
    Return an enum member by index name.
    """
    enum_cls_typ, idx = sig.args
    member = enum_cls_typ.instance_class[idx.literal_value]
    return context.get_constant_generic(builder, enum_cls_typ.dtype, member.value)