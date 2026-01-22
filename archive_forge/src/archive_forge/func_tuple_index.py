import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@overload_method(types.BaseTuple, 'index')
def tuple_index(tup, value):

    def tuple_index_impl(tup, value):
        for i in range(len(tup)):
            if tup[i] == value:
                return i
        raise ValueError('tuple.index(x): x not in tuple')
    return tuple_index_impl