import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.eq)
def charseq_eq(a, b):
    if not _same_kind(a, b):
        return
    left_code = _get_code_impl(a)
    right_code = _get_code_impl(b)
    if left_code is not None and right_code is not None:

        def eq_impl(a, b):
            n = len(a)
            if n != len(b):
                return False
            for i in range(n):
                if left_code(a, i) != right_code(b, i):
                    return False
            return True
        return eq_impl