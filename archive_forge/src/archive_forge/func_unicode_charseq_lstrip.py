import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'lstrip')
@overload_method(types.CharSeq, 'lstrip')
@overload_method(types.Bytes, 'lstrip')
def unicode_charseq_lstrip(a, chars=None):
    if isinstance(a, types.UnicodeCharSeq):
        if is_nonelike(chars):

            def impl(a, chars=None):
                return str(a).lstrip()
            return impl
        elif isinstance(chars, types.UnicodeCharSeq):

            def impl(a, chars=None):
                return str(a).lstrip(str(chars))
            return impl
        elif isinstance(chars, types.UnicodeType):

            def impl(a, chars=None):
                return str(a).lstrip(chars)
            return impl
    if isinstance(a, (types.CharSeq, types.Bytes)):
        if is_nonelike(chars):

            def impl(a, chars=None):
                return a._to_str().lstrip()._to_bytes()
            return impl
        elif isinstance(chars, (types.CharSeq, types.Bytes)):

            def impl(a, chars=None):
                return a._to_str().lstrip(chars._to_str())._to_bytes()
            return impl