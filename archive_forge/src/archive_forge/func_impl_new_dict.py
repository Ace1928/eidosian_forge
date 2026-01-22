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
@overload(new_dict)
def impl_new_dict(key, value, n_keys=0):
    """Creates a new dictionary with *key* and *value* as the type
    of the dictionary key and value, respectively. *n_keys* is the
    number of keys to insert without requiring a resize, where a
    value of 0 creates a dictionary with minimum size.
    """
    if any([not isinstance(key, Type), not isinstance(value, Type)]):
        raise TypeError('expecting *key* and *value* to be a numba Type')
    keyty, valty = (key, value)

    def imp(key, value, n_keys=0):
        if n_keys < 0:
            raise RuntimeError('expecting *n_keys* to be >= 0')
        dp = _dict_new_sized(n_keys, keyty, valty)
        _dict_set_method_table(dp, keyty, valty)
        d = _make_dict(keyty, valty, dp)
        return d
    return imp