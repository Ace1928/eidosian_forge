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
@overload_method(types.UnicodeType, 'istitle')
def unicode_istitle(data):
    """
    Implements UnicodeType.istitle()
    The algorithm is an approximate translation from CPython:
    https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodeobject.c#L11829-L11885 # noqa: E501
    """

    def impl(data):
        length = len(data)
        if length == 1:
            char = _get_code_point(data, 0)
            return _PyUnicode_IsUppercase(char) or _PyUnicode_IsTitlecase(char)
        if length == 0:
            return False
        cased = False
        previous_is_cased = False
        for idx in range(length):
            char = _get_code_point(data, idx)
            if _PyUnicode_IsUppercase(char) or _PyUnicode_IsTitlecase(char):
                if previous_is_cased:
                    return False
                previous_is_cased = True
                cased = True
            elif _PyUnicode_IsLowercase(char):
                if not previous_is_cased:
                    return False
                previous_is_cased = True
                cased = True
            else:
                previous_is_cased = False
        return cased
    return impl