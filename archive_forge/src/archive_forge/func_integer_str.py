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
@overload_method(types.Integer, '__str__')
def integer_str(n):
    ten = n(10)

    def impl(n):
        flag = False
        if n < 0:
            n = -n
            flag = True
        if n == 0:
            return '0'
        length = flag + 1 + int(np.floor(np.log10(n)))
        kind = PY_UNICODE_1BYTE_KIND
        char_width = _kind_to_byte_width(kind)
        s = _malloc_string(kind, char_width, length, True)
        if flag:
            _set_code_point(s, 0, ord('-'))
        idx = length - 1
        while n > 0:
            n, digit = divmod(n, ten)
            c = ord('0') + digit
            _set_code_point(s, idx, c)
            idx -= 1
        return s
    return impl