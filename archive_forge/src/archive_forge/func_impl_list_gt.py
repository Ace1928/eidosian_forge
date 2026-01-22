import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@overload(operator.gt)
def impl_list_gt(a, b):
    if not all_list(a, b):
        return

    def list_gt_impl(a, b):
        return b < a
    return list_gt_impl