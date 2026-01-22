import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.CharSeq, '_to_str')
@overload_method(types.Bytes, '_to_str')
def charseq_to_str_mth(s):
    """Convert bytes array item or bytes instance to UTF-8 str.

    Note: The usage of _to_str method can be eliminated once all
    Python bytes operations are implemented for numba Bytes objects.
    """
    get_code = _get_code_impl(s)

    def tostr_impl(s):
        n = len(s)
        is_ascii = s.isascii()
        result = unicode._empty_string(unicode.PY_UNICODE_1BYTE_KIND, n, is_ascii)
        for i in range(n):
            code = get_code(s, i)
            unicode._set_code_point(result, i, code)
        return result
    return tostr_impl