import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
@classmethod
def from_range_state(cls, context, builder, state):
    """
            Create a RangeIter initialized from the given RangeState *state*.
            """
    self = cls(context, builder)
    start = state.start
    stop = state.stop
    step = state.step
    startptr = cgutils.alloca_once(builder, start.type)
    builder.store(start, startptr)
    countptr = cgutils.alloca_once(builder, start.type)
    self.iter = startptr
    self.stop = stop
    self.step = step
    self.count = countptr
    diff = builder.sub(stop, start)
    zero = context.get_constant(int_type, 0)
    one = context.get_constant(int_type, 1)
    pos_diff = builder.icmp_signed('>', diff, zero)
    pos_step = builder.icmp_signed('>', step, zero)
    sign_differs = builder.xor(pos_diff, pos_step)
    zero_step = builder.icmp_unsigned('==', step, zero)
    with cgutils.if_unlikely(builder, zero_step):
        context.call_conv.return_user_exc(builder, ValueError, ('range() arg 3 must not be zero',))
    with builder.if_else(sign_differs) as (then, orelse):
        with then:
            builder.store(zero, self.count)
        with orelse:
            rem = builder.srem(diff, step)
            rem = builder.select(pos_diff, rem, builder.neg(rem))
            uneven = builder.icmp_signed('>', rem, zero)
            newcount = builder.add(builder.sdiv(diff, step), builder.select(uneven, one, zero))
            builder.store(newcount, self.count)
    return self