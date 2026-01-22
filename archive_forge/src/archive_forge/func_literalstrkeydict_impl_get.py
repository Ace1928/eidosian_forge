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
@overload(operator.getitem)
@overload_method(types.LiteralStrKeyDict, 'get')
def literalstrkeydict_impl_get(dct, *args):
    if not isinstance(dct, types.LiteralStrKeyDict):
        return
    msg = 'Cannot get{item}() on a literal dictionary, return type cannot be statically determined'
    raise TypingError(msg)