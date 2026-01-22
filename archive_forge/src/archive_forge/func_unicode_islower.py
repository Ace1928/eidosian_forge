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
@overload_method(types.UnicodeType, 'islower')
def unicode_islower(data):
    """
    impl is an approximate translation of:
    https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Objects/unicodeobject.c#L11900-L11933    # noqa: E501
    mixed with:
    https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Objects/bytes_methods.c#L131-L156    # noqa: E501
    """

    def impl(data):
        length = len(data)
        if length == 1:
            return _PyUnicode_IsLowercase(_get_code_point(data, 0))
        if length == 0:
            return False
        cased = False
        for idx in range(length):
            cp = _get_code_point(data, idx)
            if _PyUnicode_IsUppercase(cp) or _PyUnicode_IsTitlecase(cp):
                return False
            elif not cased and _PyUnicode_IsLowercase(cp):
                cased = True
        return cased
    return impl