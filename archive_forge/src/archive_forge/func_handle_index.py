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
@register_jitable
def handle_index(l, index):
    """Handle index.

    If the index is negative, convert it. If the index is out of range, raise
    an IndexError.
    """
    index = fix_index(l, index)
    if index < 0 or index >= len(l):
        raise IndexError('list index out of range')
    return index