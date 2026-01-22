import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
def tostr_impl(s):
    n = len(s)
    is_ascii = s.isascii()
    result = unicode._empty_string(unicode.PY_UNICODE_1BYTE_KIND, n, is_ascii)
    for i in range(n):
        code = get_code(s, i)
        unicode._set_code_point(result, i, code)
    return result