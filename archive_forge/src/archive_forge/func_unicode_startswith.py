import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@overload_method(types.UnicodeType, 'startswith')
def unicode_startswith(s, prefix, start=None, end=None):
    if not is_nonelike(start) and (not isinstance(start, types.Integer)):
        raise TypingError("When specified, the arg 'start' must be an Integer or None")
    if not is_nonelike(end) and (not isinstance(end, types.Integer)):
        raise TypingError("When specified, the arg 'end' must be an Integer or None")
    if isinstance(prefix, types.UniTuple) and isinstance(prefix.dtype, types.UnicodeType):

        def startswith_tuple_impl(s, prefix, start=None, end=None):
            for item in prefix:
                if s.startswith(item, start, end):
                    return True
            return False
        return startswith_tuple_impl
    elif isinstance(prefix, types.UnicodeCharSeq):

        def startswith_char_seq_impl(s, prefix, start=None, end=None):
            return s.startswith(str(prefix), start, end)
        return startswith_char_seq_impl
    elif isinstance(prefix, types.UnicodeType):

        def startswith_unicode_impl(s, prefix, start=None, end=None):
            length, prefix_length = (len(s), len(prefix))
            if start is None:
                start = 0
            if end is None:
                end = length
            start, end = _adjust_indices(length, start, end)
            if end - start < prefix_length:
                return False
            if prefix_length == 0:
                return True
            s_slice = s[start:end]
            return _cmp_region(s_slice, 0, prefix, 0, prefix_length) == 0
        return startswith_unicode_impl
    else:
        raise TypingError("The arg 'prefix' should be a string or a tuple of strings")