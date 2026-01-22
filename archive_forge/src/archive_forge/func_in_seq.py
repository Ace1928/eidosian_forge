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
@lower_builtin(operator.contains, types.Sequence, types.Any)
def in_seq(context, builder, sig, args):

    def seq_contains_impl(lst, value):
        for elem in lst:
            if elem == value:
                return True
        return False
    return context.compile_internal(builder, seq_contains_impl, sig, args)