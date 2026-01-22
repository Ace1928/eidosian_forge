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
@overload_method(types.ListType, 'extend')
def impl_extend(l, iterable):
    if not isinstance(l, types.ListType):
        return
    if not isinstance(iterable, types.IterableType):
        raise TypingError('extend argument must be iterable')
    _check_for_none_typed(l, 'extend')

    def select_impl():
        if isinstance(iterable, types.ListType):

            def impl(l, iterable):
                if not l._is_mutable():
                    raise ValueError('list is immutable')
                if l is iterable:
                    iterable = iterable.copy()
                for i in iterable:
                    l.append(i)
            return impl
        else:

            def impl(l, iterable):
                for i in iterable:
                    l.append(i)
            return impl
    if l.is_precise():
        return select_impl()
    else:
        if hasattr(iterable, 'dtype'):
            ty = iterable.dtype
        elif hasattr(iterable, 'item_type'):
            ty = iterable.item_type
        elif hasattr(iterable, 'yield_type'):
            ty = iterable.yield_type
        elif isinstance(iterable, types.UnicodeType):
            ty = iterable
        else:
            raise TypingError('unable to extend list, iterable is missing either *dtype*, *item_type* or *yield_type*.')
        l = l.refine(ty)
        sig = typing.signature(types.void, l, iterable)
        return (sig, select_impl())