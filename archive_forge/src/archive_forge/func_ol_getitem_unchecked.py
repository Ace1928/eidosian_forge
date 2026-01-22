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
@overload_method(types.ListType, 'getitem_unchecked')
def ol_getitem_unchecked(lst, index):
    if not isinstance(index, types.Integer):
        return

    def impl(lst, index):
        index = fix_index(lst, index)
        castedindex = _cast(index, types.intp)
        _, item = _list_getitem(lst, castedindex)
        return _nonoptional(item)
    return impl