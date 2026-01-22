import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
def getitem_impl(s, i):
    if i < max_i and i >= 0:
        return get_value(s, i)
    raise IndexError(msg)