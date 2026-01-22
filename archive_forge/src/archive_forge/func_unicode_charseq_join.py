import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'join')
@overload_method(types.CharSeq, 'join')
@overload_method(types.Bytes, 'join')
def unicode_charseq_join(a, parts):
    if isinstance(a, types.UnicodeCharSeq):

        def impl(a, parts):
            _parts = [str(p) for p in parts]
            return str(a).join(_parts)
        return impl
    if isinstance(a, (types.CharSeq, types.Bytes)):

        def impl(a, parts):
            _parts = [p._to_str() for p in parts]
            return a._to_str().join(_parts)._to_bytes()
        return impl