import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.add)
@overload(operator.iadd)
def charseq_concat(a, b):
    if not _same_kind(a, b):
        return
    if isinstance(a, types.UnicodeCharSeq) and isinstance(b, types.UnicodeType):

        def impl(a, b):
            return str(a) + b
        return impl
    if isinstance(b, types.UnicodeCharSeq) and isinstance(a, types.UnicodeType):

        def impl(a, b):
            return a + str(b)
        return impl
    if isinstance(a, types.UnicodeCharSeq) and isinstance(b, types.UnicodeCharSeq):

        def impl(a, b):
            return str(a) + str(b)
        return impl
    if isinstance(a, (types.CharSeq, types.Bytes)) and isinstance(b, (types.CharSeq, types.Bytes)):

        def impl(a, b):
            return (a._to_str() + b._to_str())._to_bytes()
        return impl