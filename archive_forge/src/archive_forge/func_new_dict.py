import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
def new_dict(key, value, n_keys=0):
    """Construct a new dict with enough space for *n_keys* without a resize.

    Parameters
    ----------
    key, value : TypeRef
        Key type and value type of the new dict.
    n_keys : int, default 0
        The number of keys to insert without needing a resize.
        A value of 0 creates a dict with minimum size.
    """
    return dict()