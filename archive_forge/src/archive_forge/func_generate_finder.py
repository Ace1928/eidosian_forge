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
def generate_finder(find_func):
    """Generate finder either left or right."""

    def impl(data, substr, start=None, end=None):
        length = len(data)
        sub_length = len(substr)
        if start is None:
            start = 0
        if end is None:
            end = length
        start, end = _adjust_indices(length, start, end)
        if end - start < sub_length:
            return -1
        return find_func(data, substr, start, end)
    return impl