import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_cast(types.IntEnumMember, types.Integer)
def int_enum_to_int(context, builder, fromty, toty, val):
    """
    Convert an IntEnum member to its raw integer value.
    """
    return context.cast(builder, val, fromty.dtype, toty)