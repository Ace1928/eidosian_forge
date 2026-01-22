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
@overload_method(types.UnicodeType, 'center')
def unicode_center(string, width, fillchar=' '):
    if not isinstance(width, types.Integer):
        raise TypingError('The width must be an Integer')
    if isinstance(fillchar, types.UnicodeCharSeq):

        def center_impl(string, width, fillchar=' '):
            return string.center(width, str(fillchar))
        return center_impl
    if not (fillchar == ' ' or isinstance(fillchar, (types.Omitted, types.UnicodeType))):
        raise TypingError('The fillchar must be a UnicodeType')

    def center_impl(string, width, fillchar=' '):
        str_len = len(string)
        fillchar_len = len(fillchar)
        if fillchar_len != 1:
            raise ValueError('The fill character must be exactly one character long')
        if width <= str_len:
            return string
        allmargin = width - str_len
        lmargin = allmargin // 2 + (allmargin & width & 1)
        rmargin = allmargin - lmargin
        l_string = fillchar * lmargin
        if lmargin == rmargin:
            return l_string + string + l_string
        else:
            return l_string + string + fillchar * rmargin
    return center_impl