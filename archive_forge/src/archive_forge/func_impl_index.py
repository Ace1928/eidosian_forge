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
@overload_method(types.ListType, 'index')
def impl_index(l, item, start=None, end=None):
    if not isinstance(l, types.ListType):
        return
    _check_for_none_typed(l, 'index')
    itemty = l.item_type

    def check_arg(arg, name):
        if not (arg is None or arg in index_types or isinstance(arg, (types.Omitted, types.NoneType))):
            raise TypingError('{} argument for index must be an integer'.format(name))
    check_arg(start, 'start')
    check_arg(end, 'end')

    def impl(l, item, start=None, end=None):
        casteditem = _cast(item, itemty)
        for i in handle_slice(l, slice(start, end, 1)):
            if l[i] == casteditem:
                return i
        else:
            raise ValueError('item not in list')
    return impl