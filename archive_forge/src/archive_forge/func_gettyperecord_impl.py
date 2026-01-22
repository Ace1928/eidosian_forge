from collections import namedtuple
from enum import IntEnum
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.imputils import (impl_ret_untracked)
from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError
@overload(_PyUnicode_gettyperecord)
def gettyperecord_impl(a):
    """
    Provides a _PyUnicode_gettyperecord binding, for convenience it will accept
    single character strings and code points.
    """
    if isinstance(a, types.UnicodeType):
        from numba.cpython.unicode import _get_code_point

        def impl(a):
            if len(a) > 1:
                msg = 'gettyperecord takes a single unicode character'
                raise ValueError(msg)
            code_point = _get_code_point(a, 0)
            data = _gettyperecord_impl(_Py_UCS4(code_point))
            return data
        return impl
    if isinstance(a, types.Integer):
        return lambda a: _gettyperecord_impl(_Py_UCS4(a))