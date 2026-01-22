import operator
from enum import IntEnum
from llvmlite import ir
from numba.core.extending import (
from numba.core.imputils import iternext_impl
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
from numba.cpython import listobj
@overload_method(types.ListType, '_make_immutable')
def impl_make_immutable(l):
    """list._make_immutable()"""
    if isinstance(l, types.ListType):

        def impl(l):
            _list_set_is_mutable(l, 0)
        return impl