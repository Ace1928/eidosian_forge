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
@lower_cast(types.LiteralList, types.LiteralList)
def literallist_to_literallist(context, builder, fromty, toty, val):
    if len(fromty) != len(toty):
        raise NotImplementedError
    olditems = cgutils.unpack_tuple(builder, val, len(fromty))
    items = [context.cast(builder, v, f, t) for v, f, t in zip(olditems, fromty, toty)]
    return context.make_tuple(builder, toty, items)