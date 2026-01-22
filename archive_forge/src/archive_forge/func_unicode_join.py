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
@overload_method(types.UnicodeType, 'join')
def unicode_join(sep, parts):
    if isinstance(parts, types.List):
        if isinstance(parts.dtype, types.UnicodeType):

            def join_list_impl(sep, parts):
                return join_list(sep, parts)
            return join_list_impl
        elif isinstance(parts.dtype, types.UnicodeCharSeq):

            def join_list_impl(sep, parts):
                _parts = [str(p) for p in parts]
                return join_list(sep, _parts)
            return join_list_impl
        else:
            pass
    elif isinstance(parts, types.IterableType):

        def join_iter_impl(sep, parts):
            parts_list = [p for p in parts]
            return sep.join(parts_list)
        return join_iter_impl
    elif isinstance(parts, types.UnicodeType):

        def join_str_impl(sep, parts):
            parts_list = [parts[i] for i in range(len(parts))]
            return join_list(sep, parts_list)
        return join_str_impl