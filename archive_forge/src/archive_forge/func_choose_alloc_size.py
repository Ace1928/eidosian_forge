import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@classmethod
def choose_alloc_size(cls, context, builder, nitems):
    """
        Choose a suitable number of entries for the given number of items.
        """
    intp_t = nitems.type
    one = ir.Constant(intp_t, 1)
    minsize = ir.Constant(intp_t, MINSIZE)
    min_entries = builder.shl(nitems, one)
    size_p = cgutils.alloca_once_value(builder, minsize)
    bb_body = builder.append_basic_block('calcsize.body')
    bb_end = builder.append_basic_block('calcsize.end')
    builder.branch(bb_body)
    with builder.goto_block(bb_body):
        size = builder.load(size_p)
        is_large_enough = builder.icmp_unsigned('>=', size, min_entries)
        with builder.if_then(is_large_enough, likely=False):
            builder.branch(bb_end)
        next_size = builder.shl(size, one)
        builder.store(next_size, size_p)
        builder.branch(bb_body)
    builder.position_at_end(bb_end)
    return builder.load(size_p)