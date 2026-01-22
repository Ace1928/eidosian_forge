from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
def get_slice_length(builder, slicestruct):
    """
    Given a slice, compute the number of indices it spans, i.e. the
    number of iterations that for_range_slice() will execute.

    Pseudo-code:
        assert step != 0
        if step > 0:
            if stop <= start:
                return 0
            else:
                return (stop - start - 1) // step + 1
        else:
            if stop >= start:
                return 0
            else:
                return (stop - start + 1) // step + 1

    (see PySlice_GetIndicesEx() in CPython)
    """
    start = slicestruct.start
    stop = slicestruct.stop
    step = slicestruct.step
    one = ir.Constant(start.type, 1)
    zero = ir.Constant(start.type, 0)
    is_step_negative = cgutils.is_neg_int(builder, step)
    delta = builder.sub(stop, start)
    pos_dividend = builder.sub(delta, one)
    neg_dividend = builder.add(delta, one)
    dividend = builder.select(is_step_negative, neg_dividend, pos_dividend)
    nominal_length = builder.add(one, builder.sdiv(dividend, step))
    is_zero_length = builder.select(is_step_negative, builder.icmp_signed('>=', delta, zero), builder.icmp_signed('<=', delta, zero))
    return builder.select(is_zero_length, zero, nominal_length)