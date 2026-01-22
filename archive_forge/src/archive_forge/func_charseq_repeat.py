import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.mul)
def charseq_repeat(a, b):
    if isinstance(a, types.UnicodeCharSeq):

        def wrap(a, b):
            return str(a) * b
        return wrap
    if isinstance(b, types.UnicodeCharSeq):

        def wrap(a, b):
            return a * str(b)
        return wrap
    if isinstance(a, (types.CharSeq, types.Bytes)):

        def wrap(a, b):
            return (a._to_str() * b)._to_bytes()
        return wrap
    if isinstance(b, (types.CharSeq, types.Bytes)):

        def wrap(a, b):
            return (a * b._to_str())._to_bytes()
        return wrap