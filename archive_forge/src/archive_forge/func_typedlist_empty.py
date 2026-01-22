from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
@overload_classmethod(ListType, 'empty_list')
def typedlist_empty(cls, item_type, allocated=DEFAULT_ALLOCATED):
    if cls.instance_type is not ListType:
        return

    def impl(cls, item_type, allocated=DEFAULT_ALLOCATED):
        return listobject.new_list(item_type, allocated=allocated)
    return impl