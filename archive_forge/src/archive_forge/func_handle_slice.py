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
def handle_slice(l, s):
    """Handle slice.

    Convert a slice object for a given list into a range object that can be
    used to index the list. Many subtle caveats here, especially if the step is
    negative.
    """
    if len(l) == 0:
        return range(0)
    ll, sa, so, se = (len(l), s.start, s.stop, s.step)
    if se > 0:
        start = max(ll + sa, 0) if s.start < 0 else min(ll, sa)
        stop = max(ll + so, 0) if so < 0 else min(ll, so)
    elif se < 0:
        start = max(ll + sa, -1) if s.start < 0 else min(ll - 1, sa)
        stop = max(ll + so, -1) if so < 0 else min(ll, so)
    else:
        raise ValueError('slice step cannot be zero')
    return range(start, stop, s.step)