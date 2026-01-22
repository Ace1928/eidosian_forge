from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
@lower_constant(types.SliceType)
def constant_slice(context, builder, ty, pyval):
    if isinstance(ty, types.Literal):
        typ = ty.literal_type
    else:
        typ = ty
    return make_slice_from_constant(context, builder, typ, pyval)