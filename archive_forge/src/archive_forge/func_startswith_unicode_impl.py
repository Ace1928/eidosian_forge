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