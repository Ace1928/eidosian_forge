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
def split_whitespace_impl(a, sep=None, maxsplit=-1):
    a_len = len(a)
    parts = []
    last = 0
    idx = 0
    split_count = 0
    in_whitespace_block = True
    for idx in range(a_len):
        code_point = _get_code_point(a, idx)
        is_whitespace = _PyUnicode_IsSpace(code_point)
        if in_whitespace_block:
            if is_whitespace:
                pass
            else:
                last = idx
                in_whitespace_block = False
        elif not is_whitespace:
            pass
        else:
            parts.append(a[last:idx])
            in_whitespace_block = True
            split_count += 1
            if maxsplit != -1 and split_count == maxsplit:
                break
    if last <= a_len and (not in_whitespace_block):
        parts.append(a[last:])
    return parts