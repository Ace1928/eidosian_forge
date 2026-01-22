import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'rfind')
@overload_method(types.CharSeq, 'rfind')
@overload_method(types.Bytes, 'rfind')
def unicode_charseq_rfind(a, b):
    if isinstance(a, types.UnicodeCharSeq):
        if isinstance(b, types.UnicodeCharSeq):

            def impl(a, b):
                return str(a).rfind(str(b))
            return impl
        if isinstance(b, types.UnicodeType):

            def impl(a, b):
                return str(a).rfind(b)
            return impl
    if isinstance(a, types.CharSeq):
        if isinstance(b, (types.CharSeq, types.Bytes)):

            def impl(a, b):
                return a._to_str().rfind(b._to_str())
            return impl
    if isinstance(a, types.UnicodeType):
        if isinstance(b, types.UnicodeCharSeq):

            def impl(a, b):
                return a.rfind(str(b))
            return impl
    if isinstance(a, types.Bytes):
        if isinstance(b, types.CharSeq):

            def impl(a, b):
                return a._to_str().rfind(b._to_str())
            return impl