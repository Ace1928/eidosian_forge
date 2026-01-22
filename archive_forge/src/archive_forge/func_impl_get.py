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
@overload_method(types.DictType, 'get')
def impl_get(dct, key, default=None):
    if not isinstance(dct, types.DictType):
        return
    keyty = dct.key_type
    valty = dct.value_type
    _sentry_safe_cast_default(default, valty)

    def impl(dct, key, default=None):
        castedkey = _cast(key, keyty)
        ix, val = _dict_lookup(dct, castedkey, hash(castedkey))
        if ix > DKIX.EMPTY:
            return val
        return default
    return impl