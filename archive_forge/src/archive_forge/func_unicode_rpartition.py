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
@overload_method(types.UnicodeType, 'rpartition')
def unicode_rpartition(data, sep):
    """Implements str.rpartition()"""
    thety = sep
    if isinstance(sep, types.Omitted):
        thety = sep.value
    elif isinstance(sep, types.Optional):
        thety = sep.type
    accepted = (types.UnicodeType, types.UnicodeCharSeq)
    if thety is not None and (not isinstance(thety, accepted)):
        msg = '"{}" must be {}, not {}'.format('sep', accepted, sep)
        raise TypingError(msg)

    def impl(data, sep):
        sep = str(sep)
        empty_str = _empty_string(data._kind, 0, data._is_ascii)
        sep_length = len(sep)
        if data._kind < sep._kind or len(data) < sep_length:
            return (empty_str, empty_str, data)
        if sep_length == 0:
            raise ValueError('empty separator')
        pos = data.rfind(sep)
        if pos < 0:
            return (empty_str, empty_str, data)
        return (data[0:pos], sep, data[pos + sep_length:len(data)])
    return impl