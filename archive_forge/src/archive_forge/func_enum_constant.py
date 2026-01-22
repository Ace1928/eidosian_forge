import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_constant(types.EnumMember)
def enum_constant(context, builder, ty, pyval):
    """
    Return a LLVM constant representing enum member *pyval*.
    """
    return context.get_constant_generic(builder, ty.dtype, pyval.value)