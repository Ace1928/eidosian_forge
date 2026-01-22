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
@lower_builtin(operator.imul, types.List, types.Integer)
def list_mul_inplace(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    src_size = inst.size
    mult = args[1]
    zero = ir.Constant(mult.type, 0)
    mult = builder.select(cgutils.is_neg_int(builder, mult), zero, mult)
    nitems = builder.mul(mult, src_size)
    inst.resize(nitems)
    with cgutils.for_range_slice(builder, src_size, nitems, src_size, inc=True) as (dest_offset, _):
        with cgutils.for_range(builder, src_size) as loop:
            value = inst.getitem(loop.index)
            inst.setitem(builder.add(loop.index, dest_offset), value, incref=True)
    return impl_ret_borrowed(context, builder, sig.return_type, inst.value)