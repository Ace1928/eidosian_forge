from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
def make_slice_from_constant(context, builder, ty, pyval):
    sli = context.make_helper(builder, ty)
    lty = context.get_value_type(types.intp)
    default_start_pos, default_start_neg, default_stop_pos, default_stop_neg, default_step = [context.get_constant(types.intp, x) for x in get_defaults(context)]
    step = pyval.step
    if step is None:
        step_is_neg = False
        step = default_step
    else:
        step_is_neg = step < 0
        step = lty(step)
    start = pyval.start
    if start is None:
        if step_is_neg:
            start = default_start_neg
        else:
            start = default_start_pos
    else:
        start = lty(start)
    stop = pyval.stop
    if stop is None:
        if step_is_neg:
            stop = default_stop_neg
        else:
            stop = default_stop_pos
    else:
        stop = lty(stop)
    sli.start = start
    sli.stop = stop
    sli.step = step
    return sli._getvalue()