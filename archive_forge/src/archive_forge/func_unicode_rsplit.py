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
@overload_method(types.UnicodeType, 'rsplit')
def unicode_rsplit(data, sep=None, maxsplit=-1):
    """Implements str.unicode_rsplit()"""

    def _unicode_rsplit_check_type(ty, name, accepted):
        """Check object belongs to one of specified types"""
        thety = ty
        if isinstance(ty, types.Omitted):
            thety = ty.value
        elif isinstance(ty, types.Optional):
            thety = ty.type
        if thety is not None and (not isinstance(thety, accepted)):
            raise TypingError('"{}" must be {}, not {}'.format(name, accepted, ty))
    _unicode_rsplit_check_type(sep, 'sep', (types.UnicodeType, types.UnicodeCharSeq, types.NoneType))
    _unicode_rsplit_check_type(maxsplit, 'maxsplit', (types.Integer, int))
    if sep is None or isinstance(sep, (types.NoneType, types.Omitted)):

        def rsplit_whitespace_impl(data, sep=None, maxsplit=-1):
            if data._is_ascii:
                return ascii_rsplit_whitespace_impl(data, sep, maxsplit)
            return unicode_rsplit_whitespace_impl(data, sep, maxsplit)
        return rsplit_whitespace_impl

    def rsplit_impl(data, sep=None, maxsplit=-1):
        sep = str(sep)
        if data._kind < sep._kind or len(data) < len(sep):
            return [data]

        def _rsplit_char(data, ch, maxsplit):
            result = []
            ch_code_point = _get_code_point(ch, 0)
            i = j = len(data) - 1
            while i >= 0 and maxsplit > 0:
                data_code_point = _get_code_point(data, i)
                if data_code_point == ch_code_point:
                    result.append(data[i + 1:j + 1])
                    j = i = i - 1
                    maxsplit -= 1
                i -= 1
            if j >= -1:
                result.append(data[0:j + 1])
            return result[::-1]
        if maxsplit < 0:
            maxsplit = sys.maxsize
        sep_length = len(sep)
        if sep_length == 0:
            raise ValueError('empty separator')
        if sep_length == 1:
            return _rsplit_char(data, sep, maxsplit)
        result = []
        j = len(data)
        while maxsplit > 0:
            pos = data.rfind(sep, start=0, end=j)
            if pos < 0:
                break
            result.append(data[pos + sep_length:j])
            j = pos
            maxsplit -= 1
        result.append(data[0:j])
        return result[::-1]
    return rsplit_impl