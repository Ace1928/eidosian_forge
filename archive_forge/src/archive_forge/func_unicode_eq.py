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
@overload(operator.eq)
def unicode_eq(a, b):
    if not (a.is_internal and b.is_internal):
        return
    if isinstance(a, types.Optional):
        check_a = a.type
    else:
        check_a = a
    if isinstance(b, types.Optional):
        check_b = b.type
    else:
        check_b = b
    accept = (types.UnicodeType, types.StringLiteral, types.UnicodeCharSeq)
    a_unicode = isinstance(check_a, accept)
    b_unicode = isinstance(check_b, accept)
    if a_unicode and b_unicode:

        def eq_impl(a, b):
            a_none = a is None
            b_none = b is None
            if a_none or b_none:
                if a_none and b_none:
                    return True
                else:
                    return False
            a = str(a)
            b = str(b)
            if len(a) != len(b):
                return False
            return _cmp_region(a, 0, b, 0, len(a)) == 0
        return eq_impl
    elif a_unicode ^ b_unicode:

        def eq_impl(a, b):
            return False
        return eq_impl