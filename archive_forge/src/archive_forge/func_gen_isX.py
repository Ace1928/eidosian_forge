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
def gen_isX(_PyUnicode_IS_func, empty_is_false=True):

    def unicode_isX(data):

        def impl(data):
            length = len(data)
            if length == 1:
                return _PyUnicode_IS_func(_get_code_point(data, 0))
            if empty_is_false and length == 0:
                return False
            for i in range(length):
                code_point = _get_code_point(data, i)
                if not _PyUnicode_IS_func(code_point):
                    return False
            return True
        return impl
    return unicode_isX