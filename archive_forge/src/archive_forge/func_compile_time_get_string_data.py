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
def compile_time_get_string_data(obj):
    """Get string data from a python string for use at compile-time to embed
    the string data into the LLVM module.
    """
    from ctypes import CFUNCTYPE, c_void_p, c_int, c_uint, c_ssize_t, c_ubyte, py_object, POINTER, byref
    extract_unicode_fn = c_helpers['extract_unicode']
    proto = CFUNCTYPE(c_void_p, py_object, POINTER(c_ssize_t), POINTER(c_int), POINTER(c_uint), POINTER(c_ssize_t))
    fn = proto(extract_unicode_fn)
    length = c_ssize_t()
    kind = c_int()
    is_ascii = c_uint()
    hashv = c_ssize_t()
    data = fn(obj, byref(length), byref(kind), byref(is_ascii), byref(hashv))
    if data is None:
        raise ValueError('cannot extract unicode data from the given string')
    length = length.value
    kind = kind.value
    is_ascii = is_ascii.value
    nbytes = (length + 1) * _kind_to_byte_width(kind)
    out = (c_ubyte * nbytes).from_address(data)
    return (bytes(out), length, kind, is_ascii, hashv.value)