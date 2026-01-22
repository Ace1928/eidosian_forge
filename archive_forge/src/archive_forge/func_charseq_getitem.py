import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload(operator.getitem)
def charseq_getitem(s, i):
    get_value = None
    if isinstance(i, types.Integer):
        if isinstance(s, types.CharSeq):
            get_value = charseq_get_value
        if isinstance(s, types.UnicodeCharSeq):
            get_value = unicode_charseq_get_value
    if get_value is not None:
        max_i = s.count
        msg = 'index out of range [0, %s]' % (max_i - 1)

        def getitem_impl(s, i):
            if i < max_i and i >= 0:
                return get_value(s, i)
            raise IndexError(msg)
        return getitem_impl