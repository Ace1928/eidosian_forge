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
@overload(operator.contains)
def literal_list_contains(lst, item):
    if isinstance(lst, types.LiteralList):

        def impl(lst, item):
            for val in literal_unroll(lst):
                if val == item:
                    return True
            return False
        return impl