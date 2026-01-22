import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@overload(operator.contains)
def in_seq_empty_tuple(x, y):
    if isinstance(x, types.Tuple) and (not x.types):
        return lambda x, y: False