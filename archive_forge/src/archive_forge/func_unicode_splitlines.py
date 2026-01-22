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
@overload_method(types.UnicodeType, 'splitlines')
def unicode_splitlines(data, keepends=False):
    """Implements str.splitlines()"""
    thety = keepends
    if isinstance(keepends, types.Omitted):
        thety = keepends.value
    elif isinstance(keepends, types.Optional):
        thety = keepends.type
    accepted = (types.Integer, int, types.Boolean, bool)
    if thety is not None and (not isinstance(thety, accepted)):
        raise TypingError('"{}" must be {}, not {}'.format('keepends', accepted, keepends))

    def splitlines_impl(data, keepends=False):
        if data._is_ascii:
            return _ascii_splitlines(data, keepends)
        return _unicode_splitlines(data, keepends)
    return splitlines_impl